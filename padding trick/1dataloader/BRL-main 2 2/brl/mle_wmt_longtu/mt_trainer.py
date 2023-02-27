import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append("../../../")

import torch.nn

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse as ap

from brl.envs.wmt14ende_standford_mosesbpe37k.tokenizer import *
from brl.models.transformer_binpack import *
from brl.exps.packed_dataloader import *


class Trainer:
    def __init__(self,
                 accum_interval=1,
                 saving_interval=1500,
                 max_epochs=10,
                 max_steps=100000,
                 save_dir=None,
                 pad_id=0,
                 ):
        self.tb = SummaryWriter(log_dir=save_dir, comment=print_githash())
        self.save_dir = self.tb.get_logdir()
        print(f'save at: {self.save_dir}')
        self.ckpt_dir = os.path.join(self.save_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir)
        # self.trans_dir = os.path.join(self.save_dir, 'translations')
        # os.makedirs(self.trans_dir)

        self.epoch_dict = {}
        self.train_dict = {}
        self.valid_dict = {}
        self.records = []

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.accum_interval = accum_interval
        self.saving_interval = saving_interval
        self.pad_id = pad_id

        self.num_updates = 0
        self.num_batches = 0
        self.lr = 0
        self.should_stop = False

    def fit(self,
            model,
            train_batch_samples,
            valid_samples,
            test_samples,
            train_dataloader,
            valid_dataloader,
            test_dataloader=None,
            optimizer=None):
        # maybe construct optimizer
        if optimizer is None:
            self.optim = torch.optim.Adam(model.parameters(),
                                          betas=(0.9, 0.98),
                                          eps=1e-9)
        else:
            self.optim = optimizer

        with tqdm(
                range(self.max_epochs),
                dynamic_ncols=True,
                desc='epoch',
        ) as self.epoch_tqdm:
            for _ in self.epoch_tqdm:
                torch.cuda.empty_cache()

                self.train_epoch(model, train_batch_samples, train_dataloader)  # TODO: schuffle between epoches
                # torch.cuda.empty_cache()
                # self.valid_epoch(model, valid_dataloader)
                # if test_dataloader is not None:
                #    torch.cuda.empty_cache()
                #    self.test_epoch(model, test_dataloader)
                if self.should_stop:
                    break

    def train_epoch(self, model, batch_samples, train_dataloader):
        model.train()
        with tqdm(train_dataloader,
                  dynamic_ncols=True,
                  desc=f'train {self.epoch_tqdm.n}') as self.train_tqdm:
            for i, batch in enumerate(self.train_tqdm):
                self.train_batch(model, batch_samples[i], batch)

                if self.should_stop:
                    break

    def train_batch(self, model, samples_num, batch):
        # cross-entropy minimization
        y_pred = self.forward_batch(model, batch)
        loss, acc = self.calc_loss(y_pred, samples_num, batch)

        # Lagrange-dual minimization
        # logits = model.forward_batch(batch.src_tensor, batch.trg_tensor)
        # z = batch.trg_tensor[1:]
        # loss = self.calc_value_learning_loss(logits, z)
        # acc = torch.zeros(size=(1,))  # TODO: compute real acc

        loss.backward()
        self.num_batches += 1

        self.train_dict['t/loss'] = loss.item()
        self.train_dict['t/acc'] = acc.item()

        if self.num_batches % self.accum_interval == 0:
            self.num_updates += 1
            self.lr = model.dim ** (-0.5) * min(self.num_updates ** (-0.5), self.num_updates * 4000 ** (-1.5))
            self.train_dict['t/lr'] = self.lr
            for pg in self.optim.param_groups:
                pg['lr'] = self.lr

            self.optim.step()
            self.optim.zero_grad()

        if self.num_updates == self.max_steps:
            self.should_stop = True

        # maybe save ckpt
        if self.num_updates % self.saving_interval == 0:
            self.save_ckpt(model)

        # log
        for t, v in self.train_dict.items():
            self.tb.add_scalar(t, v, self.num_updates)
        self.train_dict['t/ups'] = self.num_updates
        self.train_tqdm.set_postfix(self.train_dict)

        r = Record(self.num_updates, self.num_batches, self.epoch_tqdm.n,
                   'train', self.train_dict)
        self.records.append(r)

    def forward_batch(self, model, batch):
        x, x_mask, y, y_mask = batch.src_tensor, batch.src_mask, batch.trg_tensor[:-1], batch.trg_mask[:-1]

        mask = (x_mask, y_mask)
        # used for training
        y_pred = model.forward_batch(x, y, mask)
        return y_pred

    def calc_loss(self, y_pred, samples_num, batch):
        y_gt = batch.trg_tensor[1:]
        loss = self.label_smoothed_nll_loss(y_pred, y_gt, samples_num)
        acc = self.acc(y_pred, y_gt)
        return loss, acc

    def calc_value_learning_loss(self, Q, z):
        """
        Q: float tensor of shape (L_max + 2, sent_num, vocab_size), giving Q(s_t^i, ...) for 1<=t<=L+2, 1<=i<=sent_num
        z: int tensor of shape (L_max + 1, sent_num), giving z_t^i for 1<=t<=L+1, 1<=i<=sent_num
        """
        max_len, sent_num, vocab_size = Q.shape
        normalization_constant = sent_num * (max_len - 1)

        with torch.no_grad():
            pi = (Q / 1.0).softmax(-1)  # take soft max (with temperature) over the Q-values
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

        BQ = (r + gamma * VQ)[1:]  # BQ[t-1][i] gives BQ_t^i, for 1<=t<=L+1, 1<=i<=sent_num
        Delta = Qt - BQ  # Delta[t-1][i] gives the td-errors \Delta_t^i, for 1<=t<=L+1, 1<=i<=sent_num
        Delta = Delta * gamma[:-1]  # mask out padding actions before summing over actions
        Lagrangian = VQ[0] - Delta.sum(dim=0)
        Lagrangian = Lagrangian.sum() / normalization_constant
        # print('\nLagrangian: ', Lagrangian.item())
        return Lagrangian

    def label_smoothed_nll_loss(self, output, target, samples_num):
        lprobs = output.log_softmax(-1)
        epsilon = 0.0  # 0.1
        ignore_index = self.pad_id
        reduce = True

        # ratio
        ratio = (target.numel() / samples_num)

        # calc loss
        target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        # handle ignored index
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
        # handle reduce
        if reduce:
            nll_loss = nll_loss.mean()
            smooth_loss = smooth_loss.mean()

        eps_i = epsilon / lprobs.size(-1)
        loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
        # print('\nnll_loss: ', nll_loss.item())
        return loss

    def save_ckpt(self, model):
        ckpt = {
            'num_updates': self.num_updates,
            'num_batchs': self.num_batches,
            'num_epochs': self.epoch_tqdm.n,
            # 'optimizer': self.optim,
            'model': model,
        }
        fname = os.path.join(self.ckpt_dir, f'{self.epoch_tqdm.n}.{self.num_updates}.ckpt')
        torch.save(ckpt, fname)

    def acc(self, output, target):
        mask = target.ne(0)
        output = output.max(-1)[1]
        correct = output.eq(target)
        acc = correct.masked_select(mask).sum().item() / mask.sum().item()
        return torch.tensor(acc)

    '''
    def valid_epoch(self, model, valid_dataloader):
        model.eval()
        with tqdm(valid_dataloader,
                  dynamic_ncols=True,
                  desc=f'valid {self.epoch_tqdm.n}') as self.train_tqdm:
            with torch.no_grad():
                for batch in self.train_tqdm:
                    self.valid_batch(model, batch)

        valid_records = list(
            filter(
                lambda r: r.num_epochs == self.epoch_tqdm.n and r.log_type ==
                'valid', self.records))
        valid_losses = list(map(lambda r: r.record['v/loss'], valid_records))
        valid_accs = list(map(lambda r: r.record['v/acc'], valid_records))
        self.epoch_dict['v/loss'] = np.array(valid_losses).mean()
        self.epoch_dict['v/acc'] = np.array(valid_accs).mean()

        for t, v in self.epoch_dict.items():
            self.tb.add_scalar(t, v, self.num_updates)
        self.epoch_tqdm.set_postfix(self.epoch_dict)

    def valid_batch(self, model, batch):
        y_pred = self.forward_batch(model, batch)
        loss, acc = self.calc_loss(y_pred, batch)

        self.valid_dict['v/loss'] = loss.item()
        self.valid_dict['v/acc'] = acc.item()

        self.train_tqdm.set_postfix(self.valid_dict)

        r = Record(self.num_updates, self.num_batches, self.epoch_tqdm.n,
                   'valid', self.valid_dict)
        self.records.append(r)

    def test_epoch(self, model, test_dataloader):
        model.eval()
        with tqdm(test_dataloader, dynamic_ncols=True,
                  desc='test') as self.test_tqdm:
            src = []
            hyp = []
            ref = []
            for batch in self.test_tqdm:
                trg_idss = Dataset.unpad(batch.trg_tensor, model.trg_tok)
                trg_str = model.trg_tok.index2str(trg_idss)
                src_idss = Dataset.unpad(batch.src_tensor, model.src_tok)
                src_str = model.src_tok.index2str(src_idss)
                try:
                    hyp_str = self.test_batch(model, batch)
                except RuntimeError:
                    self.test_tqdm.write(f'got OOM error.')
                else:
                    hyp.extend(hyp_str)
                    ref.extend(trg_str)
                    src.extend(src_str)

                if self.test_tqdm.n == 3: break

        self.save_txt(src, 'src')
        self.save_txt(ref, 'ref')
        self.save_txt(hyp, 'hyp')

        bleu = sb.corpus_bleu(hyp, [ref])
        r = Record(self.num_updates, self.num_batches, self.epoch_tqdm.n,
                   'test', {'bleu': bleu.score})
        self.records.append(r)
        self.epoch_dict['test/bleu'] = bleu.score

        for t, v in self.epoch_dict.items():
            self.tb.add_scalar(t, v, self.num_updates)
        self.epoch_tqdm.set_postfix(self.epoch_dict)

    def save_txt(self, strs, prefix):
        fname = os.path.join(self.trans_dir, f'{self.epoch_tqdm.n}.{prefix}')
        with open(fname, 'wt') as f:
            f.writelines(map(lambda s: s + '\n', strs))

    def test_batch(self, model, batch):
        index = batch.src_tensor
        with torch.no_grad():
            hyp_index = model.beam_search(index, 4)
        idss = Dataset.unpad(hyp_index, model.trg_tok)
        hyp_str = model.trg_tok.index2str(idss)
        return hyp_str
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


def cmp_model(model1_fname, model2_fname, verbose=False):
    m1 = torch.load(model1_fname, map_location=torch.device('cpu'))['model']
    m2 = torch.load(model2_fname, map_location=torch.device('cpu'))['model']
    if verbose:
        print('==================')
        print('m1 (from {})'.format(model1_fname))
        print('==================')
        print(m1.__dict__)
        print('')
        print('==================')
        print('m2 (from {})'.format(model2_fname))
        print('==================')
        print(m2.__dict__)
        print('')
    return np.all([torch.all(torch.eq(x, y)) for x, y in zip(m1.parameters(), m2.parameters())])


def main(
        # data args
        src_lang='en',
        src_bpe_fname='../../envs/wmt14ende_standford_mosesbpe37k/bpe.share.37000',
        trg_lang='de',
        trg_bpe_fname='../../envs/wmt14ende_standford_mosesbpe37k/bpe.share.37000',
        src_train_fname='../../envs/wmt14ende_standford_mosesbpe37k/val.en.id',
        trg_train_fname='../../envs/wmt14ende_standford_mosesbpe37k/val.de.id',
        # model args
        dim=512,
        num_layers=6,
        num_heads=8,
        # trainer args
        train_batch_size=5000,
        accum_interval=5,
        dropout=0.1,
        max_epochs=2,
        max_steps=300,
        saving_interval=100,
        save_dir='log',
        seed=0,
        use_cpu=False,
        # unused
        valid_batch_size=5000,
        test_batch_size=1000,
        need_test=True,
        src_valid_fname='../../envs/wmt14ende_standford_mosesbpe37k/val.en.id',
        trg_valid_fname='../../envs/wmt14ende_standford_mosesbpe37k/val.de.id',
        src_test_fname='../../envs/wmt14ende_standford_mosesbpe37k/test.en.id',
        trg_test_fname='../../envs/wmt14ende_standford_mosesbpe37k/test.de.id',
        packed_type='source-target'
):
    print('check args ...')
    print('max_steps: ', max_steps)
    print('bucket_size_per_step: {}x{}={}'.format(train_batch_size, accum_interval, train_batch_size * accum_interval))

    print('setup rand seed to {} ...'.format(seed))
    torch.manual_seed(seed)
    np.random.seed(seed)

    print('prepare tokenizer ...')
    src_tok = Tokenizer(src_lang, src_bpe_fname)
    trg_tok = Tokenizer(trg_lang, trg_bpe_fname)

    print('prepare transformer model ...')
    transformer = Transformer(len(src_tok), len(trg_tok), dim, num_layers, num_heads, dropout)
    #if not use_cpu: transformer = transformer.cuda()

    print('prepare dataset ...')
    dataset = Dataset(
        src_tok=src_tok,
        trg_tok=trg_tok,
        src_train_fname=src_train_fname,
        trg_train_fname=trg_train_fname,
        src_valid_fname=src_valid_fname,
        trg_valid_fname=trg_valid_fname,
        src_test_fname=src_test_fname,
        trg_test_fname=trg_test_fname,
        packed_type=packed_type,
    )
    dataset.setup(True, True, True)

    train_batch_samples, train_loader = dataset.train_dataloader(train_batch_size, transformer.device)
    valid_batch_samples, valid_loader = dataset.valid_dataloader(valid_batch_size, transformer.device)
    test_batch_samples, test_loader = dataset.test_dataloader(test_batch_size, transformer.device)

    print('prepare trainer ...')
    trainer = Trainer(
        accum_interval=accum_interval,
        saving_interval=saving_interval,
        max_epochs=max_epochs,
        max_steps=max_steps,
        save_dir=save_dir,
        pad_id=trg_tok.pad_id,
    )

    print('start training ...')
    trainer.fit(
        transformer,
        train_batch_samples,
        valid_batch_samples,
        test_batch_samples,
        train_loader,
        valid_loader,
        test_loader
        if need_test else None,
    )


if __name__ == '__main__':
    # for debugging only
    # DONE: comment out the following code block before check in (and switch the flag between 'DONE' and 'TO-DO')
    # main(use_cpu=True)
    # print('finished.')

    parser = ap.ArgumentParser()
    # data args
    parser.add_argument('--packed_type', type=str,
                        default='source')  # source or target or source-target or max
    parser.add_argument(
        '--src_lang',
        type=str,
        default='en',
    )
    parser.add_argument(
        '--src_bpe_fname',
        type=str,
        default='../../envs/wmt14ende_standford_mosesbpe37k/bpe.share.37000',
    )
    parser.add_argument(
        '--trg_lang',
        type=str,
        default='de',
    )
    parser.add_argument(
        '--trg_bpe_fname',
        type=str,
        default='../../envs/wmt14ende_standford_mosesbpe37k/bpe.share.37000',
    )
    parser.add_argument(
        '--src_train_fname',
        type=str,
        default='../../envs/wmt14ende_standford_mosesbpe37k/train.en.id',
    )
    parser.add_argument(
        '--trg_train_fname',
        type=str,
        default='../../envs/wmt14ende_standford_mosesbpe37k/train.de.id',
    )
    parser.add_argument(
        '--src_valid_fname',
        type=str,
        default='../../envs/wmt14ende_standford_mosesbpe37k/val.en.id',
    )
    parser.add_argument(
        '--trg_valid_fname',
        type=str,
        default='../../envs/wmt14ende_standford_mosesbpe37k/val.de.id',
    )
    parser.add_argument(
        '--src_test_fname',
        type=str,
        default='../../envs/wmt14ende_standford_mosesbpe37k/test.en.id',
    )
    parser.add_argument(
        '--trg_test_fname',
        type=str,
        default='../../envs/wmt14ende_standford_mosesbpe37k/test.de.id',
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
        default=10,
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=100000,
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=5000,
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
        default='log' + str(np.random.randint(1, 10000, 1)[0]),
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
    )

    args = parser.parse_args()
    print(args)
    main(
        src_lang=args.src_lang,
        src_bpe_fname=args.src_bpe_fname,
        trg_lang=args.trg_lang,
        trg_bpe_fname=args.trg_bpe_fname,
        src_train_fname=args.src_train_fname,
        trg_train_fname=args.trg_train_fname,
        src_valid_fname=args.src_valid_fname,
        trg_valid_fname=args.trg_valid_fname,
        src_test_fname=args.src_test_fname,
        trg_test_fname=args.trg_test_fname,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        accum_interval=args.accum_interval,
        saving_interval=args.saving_interval,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        test_batch_size=args.test_batch_size,
        need_test=args.need_test,
        save_dir=args.save_dir,
        seed=args.seed,
        packed_type=args.packed_type
    )
