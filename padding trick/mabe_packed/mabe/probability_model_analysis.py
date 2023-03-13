from brl.envs.seq2seq import *
from brl.exps.mabe.search import *
from brl.exps.mabe.search import HeuristicFunction as ProbabilityModel
import torch.distributions as dis
import datetime


def evaluate(env, translator):
    hyp_strs, lprobs, info = translator.translate_corpus(env)

    print('')
    for key in info:
        print(f'{key} \t ', end='')
        print(*info[key])

    rv_lprob = RV('', save_data=True)
    for lprob in lprobs: rv_lprob.append(lprob)
    print(f'lprob   \t {rv_lprob.mean()} \t {rv_lprob}')
    print(f'#impossible\t {env.num_instances - rv_lprob.cnt} \t '
          f'{(env.num_instances - rv_lprob.cnt) / env.num_instances * 100.}%')


class Translator(object):
    def __init__(self, prob_model, name=None, check_distribution=False):
        self.prob_model = prob_model
        self.name = name if name is not None else str(self.__class__)
        self.check_distribution = check_distribution

    def translate_corpus(self, env):
        env.reset()

        hyp_strs, lprobs = [], []
        self.timers = [RV(f'step {i} (in ms)') for i in range(10)]
        print(datetime.datetime.now())

        for i in range(env.num_instances):
            env.step(action=0)  # do an arbitrary action to initialize the episode
            assert env.cur_episode_time == 1 and len(env.cur_state[1]) == 1

            hyp_str = self.translate(env)
            hyp_strs.append(hyp_str)

            lprob = self.calc_lprob(env)
            if lprob > float('-inf'): lprobs.append(lprob)  # impossible outputs are not counted in

            # logging
            if i % 100 == 0:
                print(f'{i} \t lprob = {sum(lprobs)/len(lprobs) if len(lprobs) > 0 else float("nan")} \t '
                      f'#impossible={i+1-len(lprobs)}')
                for timer in self.timers:
                    if timer.cnt > 0: print(timer, file=sys.stderr)
            if i < 3: print('Y[{}] = {}'.format(i, env.cur_state[1]))
            #if hyp_str == '': print('search outputs empty string for sentence ', i, file=sys.stderr)

        print(datetime.datetime.now())
        _,_,_,info = env.step(action=0)  # do an arbitrary action to finalize the the epoch and get the corpus-level scores

        return hyp_strs, lprobs, info

    def translate(self):
        """To be implemented by the child class"""
        pass

    def calc_lprob(self, env):
        assert env.cur_done is True
        Y = env.cur_state[1]

        lprobs = self.prob_model(env.cur_state, output_mode=self.prob_model.output_mode+'_all')
        lprobs = lprobs[:-1]  # discard the result at terminal step
        lprob_Y = 0.
        for t_minus_1, lprob in enumerate(lprobs):
            y_t = Y[t_minus_1 + 1]  # step index starts from 1, enumerate index starts from 0
            lprob_t = lprob[y_t]
            lprob_Y += float(lprob_t)

        if self.check_distribution:
            Qs = self.prob_model(env.cur_state, output_mode='logit')
            Qs = Qs[:-1]  # discard the result at terminal step
            lprob_cd = 0.
            for t_minus_1, Q in enumerate(Qs):
                y_t = Y[t_minus_1 + 1]  # step index starts from 1, enumerate index starts from 0
                lprob_t = self.prob_model.Q2logCP(Q, check_distribution=True, y_t=y_t, eos=env.EOS)[y_t]
                lprob_cd += float(lprob_t)
            assert lprob_Y == lprob_cd

        return lprob_Y


class TranslatorEmpty(Translator):
    def translate(self, env):
        """
        run one translation episode in the given environment, where the translation is always the empty string
        """
        env.step(env.EOS)
        hyp_str = ''  # [BOS, EOS]
        return hyp_str


class TranslatorReference(TranslatorEmpty):
    def translate(self, env):
        """
        run one translation episode in the given environment, where the translations are generated
        using the reference translations stored in the environment (which is cheating)
        """
        ref_str = env.ref[env.cur_sample_id]
        ref_ids = env.trg_tok.str2ids([ref_str])[0]

        # the reference translation might occasionally exceed the maximum length set by the environment.
        # instead of dramatically increasing the maximum length limit (which might affect the performance of other translators),
        # we choose to hack the system a little bit here by immediately terminating the episode, and then modifying
        # cur_state[1] to the reference sequence (which might exceed maximum length in this case)
        env.step(env.EOS)
        env.cur_state = (env.cur_state[0], ref_ids)

        return ref_str

    # legacy code
    def output_perturbed_reference(self, env, perturb):
        env.reset()
        env.step(env.BOS)
        ref_str = env.ref[env.cur_sample_id]
        ref_ids = env.trg_tok.str2ids([ref_str])[0]

        # if len(ref_ids) <= perturb:
        #    print('{} : cannot perturb {} tokens for sequence of {} tokens.'.format(env.cur_sample_id, perturb, len(ref_ids)))
        #    assert False
        perturb = min(perturb, len(ref_ids))
        perturbed_pos = np.random.choice(len(ref_ids), perturb, replace=False)
        for pos in perturbed_pos:
            tok_id = random.sample(range(env.EOS + 1, len(env.trg_tok)), 1)[0]
            ref_ids[pos] = tok_id

        for ref_id in ref_ids:
            env.step(ref_id)
        state, reward, done, info = env.step(env.EOS)
        return reward


class TranslatorSampling(TranslatorEmpty):
    def __init__(self, prob_model, name=None, check_distribution=False, beta=1.0):
        Translator.__init__(self, prob_model, name, check_distribution)
        self.beta = beta

    def translate(self, env, beta=None):
        """
        run one translation episode in the given environment, where the translations are generated using
        policy sampling. 'beta' overrides the object-wise beta setting, if specified.
        """
        if beta is None: beta = self.beta

        t, done = 0, False
        while not done:
            # timestamp = time.perf_counter()
            lprobs = self.prob_model(env.cur_state)
            # self.timers[0].append((time.perf_counter() - timestamp)*1e3)

            # timestamp = time.perf_counter()
            lprobs = lprobs / beta
            # self.timers[1].append((time.perf_counter() - timestamp)*1e3)

            # timestamp = time.perf_counter()
            # assert 0.999 < sum(probs) < 1.001  # torch.isclose(sum(probs), 1) will fail, due to pytorch precision limit ...
            # timers[2].append((time.perf_counter() - timestamp)*1000)  # this assertion is somehow very slow ...

            # timestamp = time.perf_counter()
            action = int(dis.Categorical(logits=lprobs).sample())
            # self.timers[3].append((time.perf_counter() - timestamp)*1e3)

            # timestamp = time.perf_counter()
            state, reward, done, info = env.step(action)
            t += 1
            # self.timers[4].append((time.perf_counter() - timestamp)*1e3)

        assert env.cur_done is True
        Y = env.cur_state[1]
        hyp_str = (env.trg_tok.ids2str(Y[1:-1])[0] if len(Y) > 2 else '')  # yttm.BPE cannot handle empty string
        return hyp_str


class TranslatorVBS(TranslatorEmpty):
    def __init__(self, prob_model, name=None, check_distribution=False, beam_size=4, batch_size=None):
        Translator.__init__(self, prob_model, name, check_distribution)
        self.beam_size = beam_size
        self.batch_size = batch_size

    def translate(self, env, beam_size=None):
        """
        run one translation episode in the given environment, where the translations are generated
        using vanilla beam search. 'beam_size' overrides the object-wise beam size setting, if specified.
        """
        if beam_size is None: beam_size = self.beam_size

        Y = vanilla_beam_search(env,
                                beam_size,
                                heuristic=self.prob_model,
                                timers=self.timers,
                                batch_size=self.batch_size
                                ).cur_state[1]
        for token in Y[1:]: env.step(token)
        hyp_str = (env.trg_tok.ids2str(Y[1:-1])[0] if len(Y) > 2 else '')  # yttm.BPE cannot handle empty string
        return hyp_str





if __name__ == "__main__":
    use_cpu = True if 'CUDA_VISIBLE_DEVICES' not in os.environ or int(os.environ['CUDA_VISIBLE_DEVICES']) < 0 else False
    # for debugging only
    # DONE: comment out the following code block before check in (and switch the flag between 'DONE' and 'TO-DO')
    # use_cpu = True
    if use_cpu: torch.set_num_threads(4)

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

    if len(sys.argv) > 1 and sys.argv[1] in ['wmt14', 'wmt17', 'cnndm']:
        environment = sys.argv[1]
        sys.argv = [sys.argv[0]] + sys.argv[2:]  # remove environment name from sys.argv for back-compatibility...
    else:
        environment = 'wmt14'
        # environment = 'wmt17'
        # environment = 'cnndm'

    if environment == 'wmt14':
        """
        wmt'14 en2de
        """
        env = Translation_v1(
            env_folder='../../envs/wmt14ende_standford_mosesbpe37k',
            src_fname='test/newstest2014.en',  # 'train2k.en',  # 'newstest2014_random100.en',
            ref_fname='test/newstest2014.de',  # 'train2k.de',  # 'newstest2014_random100.de',
            vocab_src_fname='test/bpe.share.37000',
            vocab_trg_fname='test/bpe.share.37000'  # use shared en-de vocab for wmt'14 data
        )
        env.reset()
        model_fname = sys.argv[1] if len(sys.argv) > 1 else '../../models/wmt14en2de_mle100k.ckpt'

    elif environment == 'wmt17':
        """
        wmt'17 zh2en
        """
        '''
        # TODO: move the following code (which generates a lower-cased reference file) to envs/wmt17zh2en_fairseq/
        # fix error on search.py: error: the following arguments are required: data
        # fairseq issue: https://github.com/pytorch/fairseq/issues/111
        # bug reason: You need to provide a path to a directory containing the dictionary (same as you gave during training)
        #sys.argv.append('../../envs/wmt17zh2en_fairseq')
        def tokenized_ref(ref_path, save_file_path, bpe_fname):
            import nltk
            from subword_nmt.apply_bpe import BPE, read_vocabulary
            import jieba
            from fairseq.data import Dictionary
            import codecs
            dictionary = Dictionary.load((os.path.join(bpe_fname, "dict.en.txt")))
            bpe = BPE(codecs.open(os.path.join(bpe_fname, "code"), encoding='utf-8'), vocab=dictionary.symbols)
            with open(ref_path, 'r', encoding="utf-8") as reader, open(save_file_path, 'w', encoding="utf-8") as writer:
                for single in reader:
                    tokens = nltk.word_tokenize(single)
                    single = " ".join(tokens).lower()
                    # single = bpe.process_line(single)
                    writer.write(single.lower() + "\n")
            print("tokenized tokenized")
        tokenized_ref('../../envs/wmt17zh2en_fairseq/dataset/test.en',
                      '../../envs/wmt17zh2en_fairseq/dataset/test.lower.en',
                      '../../envs/wmt17zh2en_fairseq/tgt_data_bin')
        '''
        env = Translation_v1(
            env_folder='../../envs/wmt17zh2en_fairseq',
            src_fname='dataset/test.zh',
            ref_fname='dataset/test.lower.en',
            vocab_src_fname='src_data_bin',
            vocab_trg_fname='tgt_data_bin'  # use shared en-de vocab for wmt'14 en-de data
        )
        env.reset()
        model_fname = sys.argv[1] if len(sys.argv) > 1 else 'wmt17zh2en_fairseq/zh2en_mle78k.ckpt'

    elif environment == 'cnndm':
        """
        cnn-dailymail summarization
        """
        '''
        # TODO: move the following code (which generates a sampled test set) to envs/cnn_summarization/
        # fix error on search.py: error: the following arguments are required: data
        # fairseq issue: https://github.com/pytorch/fairseq/issues/111
        # bug reason: You need to provide a path to a directory containing the dictionary (same as you gave during training)
        # sys.argv.append('../../envs/cnn_summarization')
        def sample_test(file_path, save_path):
            source_file_name, target_file_name = "test.source", "test.target"
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(file_path, source_file_name), "r", encoding="utf-8") as source_reader, \
                    open(os.path.join(file_path, target_file_name), 'r', encoding="utf-8") as target_reader:
                source_lines = source_reader.readlines()
                target_lines = target_reader.readlines()
                assert len(source_lines) == len(target_lines)
                total_ids = list(range(0, len(source_lines)))
                sample_ids = random.sample(total_ids, 2000)
                sample_source = [source_lines[single_id] for single_id in sample_ids]
                sample_target = [target_lines[single_id] for single_id in sample_ids]
    
            with open(os.path.join(save_path, source_file_name), 'w', encoding="utf-8") as source_writer, \
                    open(os.path.join(save_path, target_file_name), 'w', encoding="utf-8") as target_writer:
                for source_line, target_line in zip(sample_source, sample_target):
                    source_writer.write(source_line)
                    target_writer.write(target_line)
        sample_test('../../envs/cnn_summarization/cnn_dm', '../../envs/cnn_summarization/cnn_dm/sample_test')
        # TODO: copy encoder.json and vocab.bpe to envs/cnn_summarization/src_data_bin/ and envs/cnn_summarization/tgt_data_bin/
        '''
        env = Translation_v1(
            env_folder='../../envs/cnn_summarization',
            src_fname='cnn_dm/sample_test/test.source',
            ref_fname='cnn_dm/sample_test/test.target',
            vocab_src_fname='src_data_bin',
            vocab_trg_fname='tgt_data_bin',  # use shared en-de vocab for wmt'14 en-de data
            max_len=1023
        )
        env.reset()
        model_fname = sys.argv[1] if len(sys.argv) > 1 else './cnn_dailymail/cnndm_mle55k.ckpt'

    else:
        assert False, f'{environment} is unknown environment'


    model_output = sys.argv[2] if len(sys.argv) > 2 else 'logP'
    assert model_output == 'logP' or model_output == 'logDP'
    prob_model = ProbabilityModel(model_fname, output_mode=model_output, use_cpu=use_cpu)


    """
    mode: auto, empty, sample, vbs, ref
    """
    mode = sys.argv[3] if len(sys.argv) > 3 else 'auto'

    if mode == 'sample':
        beta = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    if mode == 'vbs':
        beam_size = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else None

    if mode == 'auto':
        print('\nTranslatorEmpty')
        translator = TranslatorEmpty(prob_model)
        evaluate(env, translator)

        print('\nTranslatorReference')
        translator = TranslatorReference(prob_model)
        evaluate(env, translator)

        print('\nTranslatorSampling (beta=1.0)')
        translator = TranslatorSampling(prob_model, beta=1.0)
        evaluate(env, translator)

        print('\nTranslatorSampling (beta=0.75)')
        translator = TranslatorSampling(prob_model, beta=0.75)
        evaluate(env, translator)

        print('\nTranslatorSampling (beta=0.5)')
        translator = TranslatorSampling(prob_model, beta=0.5)
        evaluate(env, translator)

        print('\nTranslatorSampling (beta=0.25)')
        translator = TranslatorSampling(prob_model, beta=0.25)
        evaluate(env, translator)

        print('\nTranslatorSampling (beta=0.01)')
        translator = TranslatorSampling(prob_model, beta=0.01)
        evaluate(env, translator)

        print('\nTranslatorVBS (bs=1)')
        translator = TranslatorVBS(prob_model, beam_size=1, batch_size=batch_size)
        evaluate(env, translator)

        print('\nTranslatorVBS (bs=2)')
        translator = TranslatorVBS(prob_model, beam_size=4, batch_size=batch_size)
        evaluate(env, translator)

        print('\nTranslatorVBS (bs=4)')
        translator = TranslatorVBS(prob_model, beam_size=16, batch_size=batch_size)
        evaluate(env, translator)

        print('\nTranslatorVBS (bs=8)')
        translator = TranslatorVBS(prob_model, beam_size=64, batch_size=batch_size)
        evaluate(env, translator)

        print('\nTranslatorVBS (bs=16)')
        translator = TranslatorVBS(prob_model, beam_size=256, batch_size=batch_size)
        evaluate(env, translator)

        print('\nTranslatorVBS (bs=32)')
        translator = TranslatorVBS(prob_model, beam_size=1024, batch_size=batch_size)
        evaluate(env, translator)

        # TODO: add curve drawing

    elif mode == 'empty':
        print('\nTranslatorEmpty')
        translator = TranslatorEmpty(prob_model)
        evaluate(env, translator)

    elif mode == 'sample':
        print('\nTranslatorSampling (beta={})'.format(beta))
        translator = TranslatorSampling(prob_model, beta=beta)
        evaluate(env, translator)

    elif mode == 'vbs':
        print('\nTranslatorVBS (bs={})'.format(beam_size))
        translator = TranslatorVBS(prob_model, beam_size=beam_size, batch_size=batch_size)
        evaluate(env, translator)

    elif mode == 'ref':
        print('\nTranslatorReference')
        translator = TranslatorReference(prob_model)
        evaluate(env, translator)

    else:
        print('mode = {auto, empty, sample, vbs, ref}')
        assert False

    print('finished')
