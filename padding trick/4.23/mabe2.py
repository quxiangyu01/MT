
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import sys
sys.path.append("../../../")
from brl.utils import *
from tqdm import tqdm
import argparse as ap
import shutil
import time
# from brl.models.transformer_brl import *
from brl.models.transformer_best import *
from mt_dataloader import *
#from brl.envs.wmt14ende_standford_mosesbpe37k.tokenizer import *
from brl.envs.wmt17zh2en_fairseq.tokenizer import *

def main(
        # data args
        src_lang='en',
        trg_lang='de',
        src_bpe_fname='../../envs/wmt14ende_standford_mosesbpe37k/bpe.share.37000',
        trg_bpe_fname='../../envs/wmt14ende_standford_mosesbpe37k/bpe.share.37000',
        src_train_fname='../../envs/wmt14ende_standford_mosesbpe37k/test.en.id',
        trg_train_fname='../../envs/wmt14ende_standford_mosesbpe37k/test.de.id',
        # model args
        dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        use_cpu=False,
        # trainer args
        bp_bucket_size=1000,
        accum_interval=5,
        max_epochs=2,
        max_steps=1000,
        saving_interval=500,
        save_dir='log',
        seed=0,
        packed_type='source-target'
):
    print('check args ...')
    print('max_steps: ', max_steps)
    print('bucket_size_per_step: {}x{}={}'.format(bp_bucket_size, accum_interval, bp_bucket_size * accum_interval))

    print('set random seed to {} ...'.format(seed))

    def deterministic(seed):
        # Set random seeds
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    deterministic(seed)

    print('prepare tokenizer ...')
    env_folder = os.path.dirname(src_train_fname)
    sys.path.append(env_folder)
    #from tokenizer import Tokenizer
    src_tok = Tokenizer(src_lang, src_bpe_fname)
    trg_tok = Tokenizer(trg_lang, trg_bpe_fname)

    print('prepare model ...')
    model = Transformer(len(src_tok), len(trg_tok), dim, num_layers, num_heads, dropout, pad_id=trg_tok.pad_id,
                        bos_id=trg_tok.bos_id, eos_id=trg_tok.eos_id)
    #if not use_cpu: model = model.cuda()

    print('prepare data ...')
    dataset = Dataset(
        src_tok=src_tok,
        trg_tok=trg_tok,
        src_train_fname=src_train_fname,
        trg_train_fname=trg_train_fname,
        packed_type=packed_type,
    )
    dataset.setup(True, False, False)

    print('prepare trainer ...')
    trainer = Trainer(
        bp_bucket_size=bp_bucket_size,
        batches_per_step=accum_interval,
        saving_interval=saving_interval,
        max_epochs=max_epochs,
        max_steps=max_steps,
        save_dir=save_dir,
        pad_id=trg_tok.pad_id,
        bos_id=trg_tok.bos_id,
    )
    train_meter = StopwatchMeter()
    print('start training...')
    train_meter.start()
    trainer.train(model, dataset, bp_bucket_size)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


class StopwatchMeter(object):
    """Computes the sum/avg duration of some event in seconds"""

    def __init__(self):
        self.reset()

    def start(self):
        self.start_time = time.time()
        t = self.convert(self.start_time)
        print('start time is:', t)

    def stop(self, n=1):
        if self.start_time is not None:
            end_time = time.time()
            delta = end_time - self.start_time
            self.sum += delta
            self.n += n
            self.start_time = None
            t = self.convert(end_time)
            print('end time is:', t)

    def convert(self, t):
        t = time.localtime(t)
        t = time.strftime("%Y-%m-%d %H:%M:%S", t)
        return t

    def reset(self):
        self.sum = 0
        self.n = 0
        self.start_time = None

    @property
    def avg(self):
        return self.sum / self.n


class Trainer:
    def __init__(self,
                 bp_bucket_size,
                 batches_per_step,
                 saving_interval,
                 max_epochs,
                 max_steps,
                 save_dir,
                 pad_id,
                 bos_id
                 ):
        self.log_dir = save_dir
        print(f'save at: {self.log_dir}')
        self.ckpt_dir = os.path.join(self.log_dir, 'checkpoints')
        if os.path.exists(self.ckpt_dir): shutil.rmtree(self.ckpt_dir)
        os.makedirs(self.ckpt_dir)

        self.train_dict = {}

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.batches_per_step = batches_per_step
        self.bp_bucket_size = bp_bucket_size
        self.saving_interval = saving_interval
        self.pad_id = pad_id
        self.bos_id = bos_id

        self.num_epochs = 0
        self.num_updates = 0
        self.num_bpbatchs = 0
        self.lr = None
        self.avg_sgdbatch_size = None
        self.actual_sgdbatch_size = 0
        # self.should_stop = False

    def train(self, model, dataset, bp_bucket_size):
        train_dataloader = dataset.train_dataloader(bp_bucket_size, model.device)
        self.optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
        self.lr = lambda model, step_num: model.dim ** (-0.5) * min(step_num ** (-0.5), step_num * 4000 ** (-1.5))
        self.avg_sgdbatch_size = dataset.avg_num_tokens_per_batch * self.batches_per_step
        print('avg_sgdbatch_size = ', self.avg_sgdbatch_size)

        with tqdm(range(self.max_epochs), dynamic_ncols=True, desc='completed epochs') as self.epochs_tqdm:
            for epoch in self.epochs_tqdm:
                # re-schuffle data between epoches
                if epoch > 0:
                    train_dataloader = dataset.train_dataloader(bp_bucket_size, model.device)
                    self.avg_sgdbatch_size = dataset.avg_num_tokens_per_batch * self.batches_per_step

                # torch.cuda.empty_cache()  # TODO: why clear cache here? why repeatedly set the model to train mode?
                # model.train()

                with tqdm(train_dataloader, dynamic_ncols=True, desc=f'epoch {self.num_epochs}') as self.bpbatchs_tqdm:
                    for bpbatch in self.bpbatchs_tqdm:
                        self.train_bpbatch(model, bpbatch)

                        if self.num_updates == self.max_steps:
                            print('finished training -- max #step reached')
                            return

                self.num_epochs += 1

        print('finished training -- max #epoch reached')

        return

    def train_bpbatch(self, model, batch):
        x, y = batch.src_tensor, batch.trg_tensor[:-1]
        y_t = batch.trg_tensor[1:]
        y_t[y_t == self.bos_id] = self.pad_id
        q = model.forward_batch(x, y)

        with_covF = False
        nll_loss = self.label_smoothed_nll_loss(q, y_t)
        if not with_covF:
            nll_loss.backward()
            loss = nll_loss
            covF_loss = torch.tensor(float('nan'))
        else:
            covF_loss = self.calc_covF(q, y_t)
            # covF_loss.backward(retain_graph=True)
            loss = nll_loss + 1 * covF_loss  # \nabla nll + 1*cov = \nabla -MABE
            loss.backward()
        '''
        # Lagrange-dual minimization
        x, y = batch.src_tensor, batch.trg_tensor
        y_t = batch.trg_tensor[1:]
        q = model.forward_batch(x, y)
        loss = self.lamin_loss(q, y_t)
        loss.backward()
        covF_loss, nll_loss = torch.tensor(float('nan')), torch.tensor(float('nan'))
        q = q[:-1]
        '''

        pad_mask = y_t.eq(self.pad_id)
        num_tokens = int(pad_mask.numel() - pad_mask.sum())
        self.actual_sgdbatch_size += num_tokens
        lr = self.lr(model, self.num_updates + 1)
        self.num_bpbatchs += 1

        # logging
        self.train_dict['loss'] = loss.item()
        self.train_dict['nll'] = nll_loss.item()
        self.train_dict['covF'] = covF_loss.item()
        self.train_dict['acc'] = self.calc_acc(q, y_t).item()
        self.train_dict['lr'] = lr
        self.train_dict['bs'] = self.actual_sgdbatch_size
        self.train_dict['step_num'] = self.num_updates + 1
        self.bpbatchs_tqdm.set_postfix(self.train_dict, refresh=False)

        if self.num_bpbatchs % self.batches_per_step == 0:
            for pg in self.optim.param_groups:
                pg[
                    'lr'] = lr  # * (25000 / self.bs)  #* (self.bucket_size * self.accum_interval / self.bs)  # compensate lr by the bucket/batch size ratio
            self.optim.step()
            self.optim.zero_grad()
            self.actual_sgdbatch_size = 0
            self.num_updates += 1

            if self.num_updates % self.saving_interval == 0:
                self.save_ckpt(model)

    def label_smoothed_nll_loss(self, output, target):
        lprobs = output.log_softmax(-1)
        target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        # print('\nnll_loss: ', nll_loss.item())

        # label smoothing
        epsilon = 0.1  # 0.0
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        smooth_loss = smooth_loss / lprobs.size(-1)
        loss = (1. - epsilon) * nll_loss + epsilon * smooth_loss

        pad_mask = target.eq(self.pad_id)
        loss.masked_fill_(pad_mask, 0.)
        loss = loss.sum() / self.avg_sgdbatch_size
        # loss = loss.mean() / (self.avg_sgdbatch_size / loss.nelement())  # to avoid loss.sum() which can be too large
        # loss = loss.mean()  # TODO: rule out padding when calculating sample size
        # loss *= pad_mask.numel() / 25000#(self.bucket_size * self.accum_interval)  # renormalize the loss by the total bucket size
        return loss

    def calc_covF(self, Q, y_t):
        # Q_t = Q.gather(dim=-1, index=y_t.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            P = Q.softmax(-1)
            EQ = (P * Q).sum(dim=-1, keepdim=True)
            A = Q - EQ

        covF_batch = (P * Q * A).sum(dim=-1)

        pad_mask = y_t.eq(self.pad_id)
        covF_batch.masked_fill_(pad_mask, 0.)
        covF = covF_batch.sum() / self.avg_sgdbatch_size
        # covF = covF_batch.mean()  # TODO: rule out padding when calculating sample size
        # covF *= pad_mask.numel() / 25000#(self.bucket_size * self.accum_interval)  # renormalize the loss by the total bucket size
        return covF

    def lamin_loss(self, Q, z):
        """
        Q: float tensor of shape (L_max + 2, sent_num, vocab_size), giving Q(s_t^i, ...) for 1<=t<=L+2, 1<=i<=sent_num
        z: int tensor of shape (L_max + 1, sent_num), giving z_t^i for 1<=t<=L+1, 1<=i<=sent_num
        """
        max_len, sent_num, vocab_size = Q.shape
        beta = 1.0  # 0.01

        with torch.no_grad():
            pi = (Q / beta).softmax(-1)  # take soft max (with temperature) over the Q-values
            # pi = Q.ge(Q.max(-1, keepdim=True)[0])  # take hard max over the Q-values

        r = torch.rand(size=(max_len, sent_num), dtype=Q.dtype, device=Q.device, requires_grad=False)
        # r = torch.zeros(size=(max_len, sent_num), dtype=Q.dtype, device=Q.device, requires_grad=False)

        gamma = torch.ones(size=(max_len, sent_num), dtype=Q.dtype, device=Q.device, requires_grad=False)
        gamma[-1] = torch.zeros(size=(sent_num,), dtype=Q.dtype, device=Q.device, requires_grad=False)
        pad_mask = z.eq(self.pad_id)
        gamma[:-1].masked_fill_(pad_mask, 0.)

        Qt = Q[:-1].gather(dim=-1, index=z.unsqueeze(-1)).squeeze(
            -1)  # Qt[t-1][i] gives Q_t^i, for 1<=t<=L+1, 1<=i<=sent_num
        # P = torch.log( torch.exp(Q).sum(dim=-1) )
        # nll_loss = (P[:-1] - Qt) * gamma[:-1]
        # nll_loss = nll_loss.sum() / normalization_constant
        # print('\nnll_loss: ', nll_loss.item())
        # return nll_loss

        VQ = (Q * pi).sum(dim=-1)  # VQ[t-1][i] gives VQ_t^i, for 1<=t<=L+2, 1<=i<=sent_num
        # equal_grad_loss = (VQ[:-1] - Qt) * gamma[:-1]
        # equal_grad_loss = equal_grad_loss.sum() / normalization_constant
        # print('\nequal_grad_loss: ', equal_grad_loss.item())
        # return equal_grad_loss

        with torch.no_grad():
            EQ = (Q * pi).sum(dim=-1, keepdim=True)
            A = Q - EQ
        VQ += 1 * (Q * pi * A).sum(dim=-1) / beta  # MABE(0)
        # VQ += 0 * (Q * pi * A).sum(dim=-1) / beta  # MABE(1)   #lamin - 1*cov = mabe + 1*cov
        # VQ += 2 * (Q * pi * A).sum(dim=-1) / beta  # MABE(-1)  #lamin + 1*cov = mabe - 1*cov
        # VQ += (-1) * (Q * pi * A).sum(dim=-1) / beta  # MABE(2)   #lamin - 2*cov = mabe + 2*cov
        # VQ += 3 * (Q * pi * A).sum(dim=-1) / beta  # MABE(-2)  #lamin + 2*cov = mabe - 2*cov

        BQ = (r + gamma * VQ)[1:]  # BQ[t-1][i] gives BQ_t^i, for 1<=t<=L+1, 1<=i<=sent_num
        Delta = Qt - BQ  # Delta[t-1][i] gives the td-errors \Delta_t^i, for 1<=t<=L+1, 1<=i<=sent_num
        Delta = Delta * gamma[:-1]  # mask out padding actions before summing over actions
        Lagrangian = VQ[0] - Delta.sum(dim=0)
        # normalization_constant = sent_num * (max_len - 1)  # TODO: rule out padding when calculating sample size
        normalization_constant = self.avg_sgdbatch_size
        Lagrangian = Lagrangian.sum() / normalization_constant
        # print('\nLagrangian: ', Lagrangian.item())
        return Lagrangian

    def save_ckpt(self, model):
        ckpt = {
            'num_epochs': self.num_epochs,
            'num_updates': self.num_updates,
            'num_batchs': self.num_bpbatchs,
            # 'optimizer': self.optim,
            'model': model,
        }
        fname = os.path.join(self.ckpt_dir, f'{self.num_epochs}.{self.num_updates}.ckpt')
        torch.save(ckpt, fname)

    def calc_acc(self, output, target):
        mask = target.ne(0)
        output = output.max(-1)[1]
        correct = output.eq(target)
        acc = correct.masked_select(mask).sum().item() / mask.sum().item()
        return torch.tensor(acc)


'''
class Record:
    __slots__ = [
        'num_updates', 'num_batchs', 'num_epochs', 'timestamp', 'log_type',
        'record'
    ]

    def __init__(self, num_updates, num_batchs, num_epochs, log_type, record):
        self.num_updates = num_updates
        self.num_batchs = num_batchs
        self.num_epochs = num_epochs
        self.timestamp = time.time()
        self.log_type = log_type
        self.record = record
'''

if __name__ == '__main__':
    # for debugging only
    # DONE: comment out the following code block before check in (and switch the flag between 'DONE' and 'TO-DO')
    # main(use_cpu=True)  # wmt14 debug setting
    ''' wmt17 debug setting
    main(
        # data args
        src_lang='zh',
        trg_lang='en',
        src_bpe_fname='../../envs/wmt17zh2en_fairseq/src_data_bin',
        trg_bpe_fname='../../envs/wmt17zh2en_fairseq/tgt_data_bin',
        src_train_fname='../../envs/wmt17zh2en_fairseq/test.zh.id',
        trg_train_fname='../../envs/wmt17zh2en_fairseq/test.en.id',
        # model args
        dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.3,
        use_cpu=True,
        # trainer args
        bp_bucket_size=1000,
        accum_interval=5,
        max_epochs=2,
        max_steps=1000,
        saving_interval=500,
        save_dir='log',
        seed=0,
    )
    '''

    parser = ap.ArgumentParser()
    # data args
    parser.add_argument('--packed_type', type=str,
                        default=None)  # source or target or source-target or max

    parser.add_argument(
        '--src_lang',
        type=str,
        default='zh',
    )
    parser.add_argument(
        '--src_bpe_fname',
        type=str,
        default='../../envs/wmt17zh2en_fairseq/src_data_bin',
    )
    parser.add_argument(
        '--trg_lang',
        type=str,
        default='en',
    )
    parser.add_argument(
        '--trg_bpe_fname',
        type=str,
        default='../../envs/wmt17zh2en_fairseq/tgt_data_bin',
    )
    parser.add_argument(
        '--src_train_fname',
        type=str,
        default='../../envs/wmt17zh2en_fairseq/train.clean.zh.id',
    )
    parser.add_argument(
        '--trg_train_fname',
        type=str,
        default='../../envs/wmt17zh2en_fairseq/train.clean.en.id',
    )
    parser.add_argument(
        '--src_valid_fname',
        type=str,
        default='../../envs/wmt17zh2en_fairseq/valid.clean.zh.id',
    )
    parser.add_argument(
        '--trg_valid_fname',
        type=str,
        default='../../envs/wmt17zh2en_fairseq/valid.clean.en.id',
    )
    parser.add_argument(
        '--src_test_fname',
        type=str,
        default='../../envs/wmt17zh2en_fairseq/test.zh.id',
    )
    parser.add_argument(
        '--trg_test_fname',
        type=str,
        default='../../envs/wmt17zh2en_fairseq/test.en.id',
    )
    # transformer args
    parser.add_argument(
        '--dim',
        type=int,
        default=512,
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=6,
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        default=8,
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
    )
    # trainer args
    parser.add_argument(
        '--accum_interval',
        type=int,
        default=2,
    )
    parser.add_argument(
        '--saving_interval',
        type=int,
        default=15,
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=2,
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=100000,
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=12500,
    )
    parser.add_argument(
        '--valid_batch_size',
        type=int,
        default=12500,
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=256,
    )
    parser.add_argument(
        '--need_test',
        default=False,
        action='store_true',
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='log',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
    )

    args = parser.parse_args()

    main(
        src_lang=args.src_lang,
        src_bpe_fname=args.src_bpe_fname,
        trg_lang=args.trg_lang,
        trg_bpe_fname=args.trg_bpe_fname,
        src_train_fname=args.src_train_fname,
        trg_train_fname=args.trg_train_fname,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        accum_interval=args.accum_interval,
        saving_interval=args.saving_interval,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        bp_bucket_size=args.train_batch_size,
        save_dir=args.save_dir,
        seed=args.seed,
        packed_type=args.packed_type
    )
