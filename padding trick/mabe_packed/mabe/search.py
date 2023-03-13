from brl.utils import *
import datetime
from brl.envs.seq2seq import *
import torch
from torch.nn.functional import normalize


class HeuristicFunction:
    def __init__(self,
                 model_fname,  # a model file saved by pytorch 1.7.1+
                 output_mode='logit',  # can be one of: 'logit', 'logP', 'logDP', 'logit_all', 'logP_all', 'logDP_all'
                 use_cpu=False
                 ):
        self.encoder_decoder = torch.load(model_fname, map_location=torch.device('cpu' if use_cpu else 'cuda'))
        if type(self.encoder_decoder) is dict: self.encoder_decoder = self.encoder_decoder['model']  # hack: for loading single-model files in longtu's code
        self.encoder_decoder = self.encoder_decoder.eval()  # disable dropout
        self.output_mode = output_mode

    def __call__(self, state, output_mode=None) -> torch.tensor:
        X, Y, t = state[0], state[1], len(state[1])
        output = self.forward_batch([X], [Y], output_mode)

        # batch size is 1 for this routine
        output_mode = self.output_mode if output_mode is None else output_mode
        if output_mode.endswith('_all'):
            return output[:, 0, :]
        else:
            return output[0]

    def forward_batch(self, Xs, Ys, output_mode=None):
        X_tensor = torch.as_tensor(Xs, device=self.encoder_decoder.device).T
        Y_tensor = torch.as_tensor(Ys, device=self.encoder_decoder.device).T
        with torch.no_grad():
            Qs = self.encoder_decoder.forward_batch(X_tensor, Y_tensor)

        output_mode = self.output_mode if output_mode is None else output_mode
        if not output_mode.endswith('_all'):
            Qs = Qs[-1]  # we only need the output for the last token

        if output_mode.startswith('logit'): return Qs
        if output_mode.startswith('logP'): return self.Q2logP(Qs)
        if output_mode.startswith('logDP'): return self.Q2logDP(Qs)
        assert False

    def Q2logP(self, Q):
        logP = Q.log_softmax(-1)
        return logP

    def Q2logDP(self, Q, check_distribution=False, y_t=None, eos=None):
        P = Q.softmax(-1)
        EQ = (P * Q).sum(-1, keepdim=True)
        A = Q - EQ
        CP = P * (1 + A)
        CP = CP.clamp(min=0., max=1.)
        DP = normalize(CP, dim=-1, p=1.)  # probability normalization
        logDP = DP.log()

        if check_distribution:
            title = "P(EOS) = {:.2e} A(EOS) = {:+.3f} DP(EOS) = {:.2e}\n" \
                    "P(y_t) = {:.2e} A(y_t) = {:+.3f} P'(y_t) = {:.2e}".format(
                P[eos],
                A[eos],
                DP[eos],
                P[y_t],
                A[y_t],
                DP[y_t]
            )
            plot = LinePlot('token', 'A/logP/logDP', title=title)
            vocab_size = P.shape[-1]
            A = np.array(A)
            A.sort()
            plot.add_line([x for x in range(vocab_size)], A, label='A')
            logp = np.array(P.log())
            logp.sort()
            plot.add_line([x for x in range(vocab_size)], logp, label='logP')
            logDP = np.array(DP.log())
            logDP.sort()
            plot.add_line([x for x in range(vocab_size)], logDP, label='logDP')
            plot.show(block=True)
        return logDP


class Node:
    def __init__(self, env, action, value):
        self.env, self.action, self.value = env, action, value

    def make_child(self, action, value):
        return Node(env=self.env, action=action, value=value)

    def __lt__(self, other):
        return self.value < other.value

    def consolidate_values(self, nodes, q_t):
        return q_t


def beam_search(original_env, beam_size, q_func, Node, timers=None, avoid_empty=False, batch_size=None):
    open_nodes = [Node(env=original_env, action=None, value=0.)]  # (X,[BOS])
    terminal_nodes = []

    t = 1
    while True:
        # timestamp = time.perf_counter()
        assert t == 1 or len(open_nodes) + len(terminal_nodes) == beam_size

        # a naive trick to mitigate the beam search curse
        if avoid_empty:
            if t == 1:  # avoid giving empty output
                true_beam_size = beam_size
                beam_size = min(8, true_beam_size)
            else:
                beam_size = true_beam_size

        candidates = Filter(max_size=beam_size)
        for node in terminal_nodes: candidates.append(node)
        # timers[0].append((time.perf_counter() - timestamp) * 1000)

        # timestamp = time.perf_counter()
        Xs = [node.env.cur_state[0] for node in open_nodes]
        Ys = [node.env.cur_state[1] for node in open_nodes]
        if batch_size is None or len(Xs) <= batch_size:
            q_t = q_func.forward_batch(Xs, Ys)
        else:
            # need to set pytorch_thread_num to 1 otherwise torch.equal(q1,q2) is False ...
            #q1 = q_func.forward_batch(Xs[0:1], Ys[0:1])[0]
            #q2 = q_func.forward_batch(Xs[0:2], Ys[0:2])[0]
            for i in range(0, len(Xs), batch_size):
                j = min(i+batch_size,len(Xs))
                Xs_batch = Xs[i:j]
                Ys_batch = Ys[i:j]
                qt_batch = q_func.forward_batch(Xs_batch, Ys_batch)
                q_t = qt_batch if i == 0 else torch.cat((q_t, qt_batch), dim=0)

        values = Node.consolidate_values(None, open_nodes, q_t)
        # timers[1].append((time.perf_counter() - timestamp) * 1000)

        # timestamp = time.perf_counter()
        top_tokens = values.flatten().topk(beam_size)
        # timers[2].append((time.perf_counter() - timestamp) * 1000)

        # timestamp = time.perf_counter()
        indices = top_tokens.indices.cpu().numpy()
        # timers[3].append((time.perf_counter() - timestamp) * 1000)

        # timestamp = time.perf_counter()
        values = top_tokens.values.cpu().numpy()
        # timers[4].append((time.perf_counter() - timestamp) * 1000)

        # timestamp = time.perf_counter()
        for i in range(beam_size):
            index = indices[i]
            num_actions = q_t.shape[-1]
            node_id = index // num_actions
            action = int(index % num_actions)
            value = float(values[i])
            child_node = open_nodes[node_id].make_child(action, value)
            candidates.append(child_node)
        # timers[5].append((time.perf_counter() - timestamp) * 1000)

        # timestamp = time.perf_counter()
        assert len(candidates) == beam_size
        open_nodes, terminal_nodes = [], []
        best_node, num_completion = None, 0
        for node in candidates:
            if node.action is not None:
                env = copy.copy(node.env)
                env.step(node.action)
                node.env, node.action = env, None

            if node.env.cur_done:
                terminal_nodes.append(node)
            else:
                open_nodes.append(node)

            # collect sufficient statistics for termination judgement
            best_node = max_(best_node, node)
            if node.env.cur_done: num_completion += 1
        # timers[6].append((time.perf_counter() - timestamp) * 1000)

        # check termination condition
        # if best_node.env.cur_done:
        if num_completion == beam_size:
            return best_node.env

        t += 1


def vanilla_beam_search(original_env, beam_size, heuristic, timers=None, avoid_empty=False, batch_size=None):
    class Node_SeqLogProb(Node):
        def consolidate_values(self, nodes, q_t):
            values_prefix = torch.tensor([[node.value] for node in nodes], device=q_t.device)
            return q_t + values_prefix

    return beam_search(original_env, beam_size, heuristic, Node_SeqLogProb, timers, avoid_empty, batch_size)


def vanilla_beam_search_slow(original_env, beam_size, heuristic, Node, timers=None):
    """
    An implementation of vanilla beam search that takes constant GPU memory regardless of the beam size.
    The search speed is thus slow when using GPU (but should be faster when using CPU)
    """
    pool = [Node(env=original_env, action=None, value=0.)]  # (X,[BOS])

    t = 1
    while True:
        # timestamp0 = time.perf_counter()
        assert t == 1 or len(pool) == beam_size
        candidates = Filter(max_size=beam_size)
        for node in pool:
            if node.env.cur_done:
                assert node.action is None
                candidates.append(node)
            else:
                # timestamp = time.perf_counter()
                top_tokens = heuristic(node.env.cur_state).topk(beam_size)
                # timers[2].append((time.perf_counter() - timestamp) * 1000)

                assert len(top_tokens.values) == beam_size
                for i in range(beam_size):
                    # timestamp = time.perf_counter()
                    lprob_t, y_t = float(top_tokens.values[i]), int(top_tokens.indices[i])
                    # timers[3].append((time.perf_counter() - timestamp) * 1000)

                    # timestamp = time.perf_counter()
                    if hasattr(node, 'nregret'):  # collect sufficient statistics for regret-based beam search
                        nregret_t = lprob_t - Node.alpha * float(top_tokens.values[0])
                        child_node = Node(env=node.env, action=y_t, nregret=node.nregret + nregret_t)
                    else:
                        child_node = Node(env=node.env, action=y_t, value=node.value + lprob_t)
                    # timers[4].append((time.perf_counter() - timestamp) * 1000)

                    candidates.append(child_node)
                    #if lprob_t == 0.: break  # all the rest child nodes will have lprob = -inf
        # timers[0].append((time.perf_counter() - timestamp0) * 1000)

        # timestamp = time.perf_counter()
        assert len(candidates) == beam_size
        pool = []
        best_node = None
        num_complete_translations = 0
        for node in candidates:
            if node.env.cur_done:
                assert node.action is None
                new_node = node
            else:
                env = copy.copy(node.env)
                env.step(node.action)
                node.env, node.action = env, None
            pool.append(node)

            # collect sufficient statistics for termination judgement
            best_node = node if best_node is None or best_node < node else best_node
            if node.env.cur_done: num_complete_translations += 1
        # timers[1].append((time.perf_counter() - timestamp) * 1000)

        # termination judgement
        if num_complete_translations == beam_size:
        #if best_node.env.cur_done:
            return best_node.env

        t += 1


def beam_search_lastQ(original_env, beam_size, heuristic):
    return beam_search(original_env, beam_size, heuristic, Node)


# TODO: legacy code, need calibration
def beam_search_with_length_penalty(original_env, beam_size, heuristic, lp_expoent):
    """
    beam search that the "Transformer paper" (https://arxiv.org/abs/1706.03762) used, in which
    the length penalty trick was borrowed from the "GNMT paper" (https://arxiv.org/abs/1609.08144)
    """
    class Node:
        def __init__(self, env, action, lprob):
            self.env, self.action, self.lprob = env, action, lprob

        def score(self):
            Y_len = len(self.env.cur_state[1])-1  # excluding BOS but including EOS
            lp = ((Y_len + 5) / 6) ** lp_expoent
            return self.lprob / lp

        def __lt__(self, other):
            return self.score() < other.score()

    return vanilla_beam_search_slow(original_env, beam_size, heuristic, Node)


# TODO: legacy code, need calibration
def beam_search_with_regret(original_env, beam_size, heuristic, alpha):
    class Node:
        alpha = None

        def __init__(self, env, action, lprob=None, nregret=0.):
            self.env, self.action, self.lprob, self.nregret = env, action, lprob, nregret

        def __lt__(self, other):
            return self.nregret < other.nregret

        # TODO: this is legacy code for regret-based search, update it to the latest interface
        def make_child(self, action, lprob_t):
            assert hasattr(node, 'nregret')  # collect sufficient statistics for regret-based beam search
            nregret_t = lprob_t - Node.alpha * float(top_tokens.values[0])
            child_node = Node(env=node.env, action=y_t, nregret=node.nregret + nregret_t)

    Node.alpha = alpha
    return vanilla_beam_search_slow(original_env, beam_size, heuristic, Node)

''' MCTS code
N = 40000

class TreeNode(object):
    def __init__(self, id, reward=float('inf'), cnt=0):
        self.id = id
        self.reward = reward
        self.cnt = cnt
        self.sqrt_cnt = 1e-10
        self.children = None

# simulated reward setting
def PlayOut(node):
    if node.id == 0:
        return 2.0
    elif node.id <10:
        return 1.0
    else:
        return 0.0

# simulated reward setting + vanilla expansion
def Expand(node):
    assert node.children is None
    if node.id == -1:
        node.children = [TreeNode(i) for i in range(N)]
        return True
    else:
        return False

# UCB selection
def Select(node):
    assert node.children is not None
    if node.cnt == 0: return node.children[0]

    baseline = 1.5 * math.sqrt(math.log(node.cnt))
    scores = [candidate.reward + baseline / candidate.sqrt_cnt for candidate in node.children]
    return node.children[np.argmax(scores)]

# vanilla update
def Update(node, reward):
    node.reward = reward if node.cnt == 0 else (node.reward * node.cnt + reward) / (node.cnt + 1)
    node.cnt += 1
    node.sqrt_cnt = math.sqrt(node.cnt)


def MCTS(root, num_rollout):
    for rollout in range(num_rollout):
        path = [root]
        while(path[-1].children is not None):
            path.append(Select(path[-1]))

        is_not_leaf = Expand(path[-1])
        if is_not_leaf:  # a variant that avoid calling pytorch in rollout 0
            path.append(Select(path[-1]))
        reward = PlayOut(path[-1])

        for node in reversed(path):
            Update(node, reward)

        #logging
        if rollout % 1000 == 0:
            print("{} : {}".format(rollout, path[-1].id))

root = TreeNode(-1)
MCTS(root, 1000000)
'''



if __name__ == "__main__":
    use_cpu = False
    # for debugging only
    # DONE: comment out the following code block before check in (and switch the flag between 'DONE' and 'TO-DO')
    use_cpu = True
    torch.set_num_threads(4)

    def deterministic(seed):
        # Set random seeds
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    deterministic(1234)

    beam_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    model_fname = sys.argv[2] if len(sys.argv) > 2 else 'wmt14en2de_brl/wmt14_mle100k.ckpt'
    print('beam size = ', beam_size)

    env = Translation_v1(
        env_folder='../../envs/wmt14ende_standford_mosesbpe37k',
        src_fname= 'newstest2014.en',  # 'train2k.en',
        ref_fname= 'newstest2014.de',  # 'train2k.de',
        vocab_src_fname='bpe.share.37000',
        vocab_trg_fname='bpe.share.37000'  # use shared en-de vocab for wmt'14 en-de data
    )
    env.reset()

    heuristic = HeuristicFunction(model_fname, use_cpu=use_cpu, output_mode='logP')

    rv_sbleu = RV('')
    #hyps = []
    hyps_file = open('vanilla_beam_search_bs'+str(beam_size)+'.de', 'w', encoding='utf-8')
    timers = [RV('step {}'.format(i)) for i in range(7)]
    print(datetime.datetime.now())
    for i in range(env.num_instances):
        env.step(action=0)  # do an arbitrary action to go across terminal step

        env = vanilla_beam_search(env, beam_size, heuristic, timers=timers)
        # env = vanilla_beam_search_slow(env, beam_size, heuristic, Node=Node, timers=timers)
        # env = beam_search_lastQ(env, beam_size, heuristic, timers=timers)
        # env = beam_search_with_length_penalty(env, beam_size, heuristic, lp_expoent=0.6)
        # env = beam_search_with_regret(env, beam_size, heuristic, alpha=1.5)

        assert env.cur_done
        rv_sbleu.append(env.cur_reward)
        Y = env.cur_state[1]

        if i % 100 == 0:
            print(f'{i}\tsbleu = {rv_sbleu.mean()}', file=sys.stderr)
            for t, timer in enumerate(timers):
                if timer.size() > 0:
                    print('timer {}\t{:.3f} us\t{:.1f}%\t{}'.format(
                        t,
                        timer.mean() * 1000,
                        timer.mean() / sum([timer.mean() for timer in timers]) * 100,
                        timer.size()
                    ))
            if timers[0].size() > 0: print('')

        if i < 3: print(f'Y[{i}] = {Y}', file=sys.stderr)

    print(datetime.datetime.now())
    _,_,_,info = env.step(action=0)

    print(f'sent-bleu \t {rv_sbleu.mean()} \t {rv_sbleu}')

    for key in info:
        print(f'{key} \t ', end='')
        print(*info[key])

    for hyp_str in env.hyp_corpus: print(hyp_str, file=hyps_file, flush=True)

    print('finished')


