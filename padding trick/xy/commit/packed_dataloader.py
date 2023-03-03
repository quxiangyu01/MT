import torch
import pandas as pd
from brl.utils import *
import time
import random


class Dataset:
    def __init__(
            self,
            src_tok,
            trg_tok,
            src_train_fname,
            trg_train_fname,
            src_valid_fname,
            trg_valid_fname,
            src_test_fname,
            trg_test_fname,
            packed_type,
    ):
        self.src_tok = src_tok
        self.trg_tok = trg_tok

        self.src_train_fname = src_train_fname
        self.trg_train_fname = trg_train_fname
        self.src_valid_fname = src_valid_fname
        self.trg_valid_fname = trg_valid_fname
        self.src_test_fname = src_test_fname
        self.trg_test_fname = trg_test_fname
        self.packed_type = packed_type

        self.max_len = 1024

    def setup(self, train=True, valid=True, test=True):
        if train:
            self.train_df = self.read_id(self.src_train_fname,
                                         self.trg_train_fname)
        if valid:
            self.valid_df = self.read_id(self.src_valid_fname,
                                         self.trg_valid_fname)
        if test:
            self.test_df = self.read_id(self.src_test_fname,
                                        self.trg_test_fname)

    def read_id(self, src_fname, trg_fname):
        src = open(src_fname).read().strip().split('\n')
        trg = open(trg_fname).read().strip().split('\n')

        src = list(map(lambda s: list(map(int, [self.src_tok.bos_id] + s.split() + [self.src_tok.eos_id])), src))
        trg = list(map(lambda s: list(map(int, [self.src_tok.bos_id] + s.split() + [self.src_tok.eos_id])), trg))

        df = pd.DataFrame({self.src_tok.lang: src, self.trg_tok.lang: trg})
        return df

    def generate_groups(self, src_lens, trg_lens, batch_size):
        """
        src_lens: src sentence lengths (list or pd.Series) including bos and eos
        trg_lens: trg sentence lengths (list or pd.Series) including bos and eos
        batch_size: number of maximum tokens in a batch
        """
        batch_idxs = []
        batch_samples = []
        batch_idx = []
        src_max = trg_max = 0
        for i, (sl, tl) in enumerate(zip(src_lens, trg_lens)):
            if sl > self.max_len or tl > self.max_len or sl > batch_size or tl > batch_size:
                continue
            src_max = max(src_max, sl)
            trg_max = max(trg_max, tl)
            batch_idx.append(i)
            if len(batch_idx) * src_max > batch_size or len(batch_idx) * trg_max > batch_size:
                batch_samples.append(len(batch_idx[:-1]) * max(src_max, trg_max))
                batch_idxs.append(batch_idx[:-1])
                batch_idx = batch_idx[-1:]
                src_max = sl
                trg_max = tl
        batch_idxs.append(batch_idx)
        batch_samples.append(len(batch_idx) * max(src_max, trg_max))
        return batch_idxs, batch_samples

    def packed_method(self, group, src_lens, trg_lens):
        src_group_lens = src_lens[group]
        trg_group_lens = trg_lens[group]
        token_max_lens = (src_group_lens.max(), trg_group_lens.max())
        if self.packed_type == 'source-target':
            token_lens_dict = {i: (s, t) for i, s, t in zip(group, src_group_lens, trg_group_lens)}
            packs = packing2D(token_lens_dict, token_max_lens)
        elif self.packed_type == 'source':
            token_lens_dict = {i: s for i, s in zip(group, src_group_lens)}
            packs = packing1D(token_lens_dict, token_max_lens[0])
        elif self.packed_type == 'target':
            token_lens_dict = {i: t for i, t in zip(group, trg_group_lens)}
            packs = packing1D(token_lens_dict, token_max_lens[1])
        elif self.packed_type == 'max':
            token_lens_dict = {i: max(s, t) for i, s, t in zip(group, src_group_lens, trg_group_lens)}
            packs = packing1D(token_lens_dict, max(token_max_lens))

        return packs

    def bin_packing(self, df, src_lens, trg_lens, batch_groups):
        def merge_sent(lens):
            mask_token = [1, -1, 2, -2, 3, -3, 5, -5, 7, -7, 11, -11, 13, -13, 17, -17, 19, -19, 23, -23, 29, -29, 31, -31, 37, -37, 41, -41, 43, -43, 47, -47, 59, -59, 61, -61, 67, -67, 71, -71, 73, -73, 79, -79, 83, -83, 89, -89, 97, -97]
            mask = []
            for i, l in enumerate(lens):
                mask.extend([mask_token[i]] * l)
            return mask

        new_df_list = []
        batch_idxs = []
        index = 0
        print('pack type', self.packed_type)
        start = time.time()

        for group in batch_groups:
            packs = self.packed_method(group, src_lens, trg_lens)
            random.shuffle(packs)
            batch_idx = []
            for pack in packs:
                p_index = list(pack.keys())
                random.shuffle(p_index)
                src_sent = df.iloc[p_index][self.src_tok.lang].values.tolist()
                trg_sent = df.iloc[p_index][self.trg_tok.lang].values.tolist()

                src_mask = merge_sent([len(sen) for sen in src_sent])
                trg_mask = merge_sent([len(sen) for sen in trg_sent])

                src_sent = [s for sen in src_sent for s in sen]
                trg_sent = [t for sen in trg_sent for t in sen]

                new_df_list.append([src_sent, trg_sent, src_mask, trg_mask])
                batch_idx.append(index)
                index += 1
            batch_idxs.append(batch_idx)
        end = time.time()
        print('time', end - start)

        new_df = pd.DataFrame(
            columns=[self.src_tok.lang, self.trg_tok.lang, 'src_mask', 'trg_mask'],
            data=new_df_list)

        return new_df, batch_idxs

    def analyze_batch_idxs(self, batch_idxs, df, verbose):

        src_lens = df[self.src_tok.lang].map(len).values
        trg_lens = df[self.trg_tok.lang].map(len).values

        if verbose:
            print('#batch : ', len(batch_idxs))

        rv_num_toks_src = RV('#token/batch in src', save_data=True)
        rv_num_sents_src = RV('#sentence/batch in src', save_data=True)
        rv_max_len_src = RV('max_len/batch in src', save_data=True)
        rv_batch_size_src = RV('batch size in src', save_data=True)
        rv_pad_rate_src = RV('padding rate in src', save_data=True)
        stats_src = [rv_num_toks_src, rv_num_sents_src, rv_max_len_src, rv_batch_size_src, rv_pad_rate_src]

        rv_num_toks_trg = RV('#token/batch in trg', save_data=True)
        rv_num_sents_trg = RV('#sentence/batch in trg', save_data=True)
        rv_max_len_trg = RV('max_len/batch in trg', save_data=True)
        rv_batch_size_trg = RV('batch size in trg', save_data=True)
        rv_pad_rate_trg = RV('padding rate in trg', save_data=True)
        stats_trg = [rv_num_toks_trg, rv_num_sents_trg, rv_max_len_trg, rv_batch_size_trg, rv_pad_rate_trg]

        def get_stats(sent_lens, src_or_trg, rv_num_toks, rv_num_sents, rv_max_len, rv_batch_size, rv_pad_rate):
            for i, batch in enumerate(batch_idxs):
                sent_lens_batch = [sent_lens[i] for i in batch]
                num_tokens = sum(sent_lens_batch)
                num_sents = len(batch)
                max_len = max(sent_lens_batch)
                num_pads = sum([max(sent_lens_batch) - len for len in sent_lens_batch])
                batch_size = sum([max(sent_lens_batch) for len in sent_lens_batch])
                if verbose:
                    print('batch {:2d} ({}) : #tok={} , #sentence={} , max_len={:3d} , #pad={} / {} = {:.1f}%'.format(
                        i, src_or_trg, num_tokens, num_sents, max_len, num_pads, batch_size,
                        num_pads / batch_size * 100.0
                    ))
                rv_num_toks.append(num_tokens)
                rv_num_sents.append(num_sents)
                rv_max_len.append(max_len)
                rv_batch_size.append(batch_size)
                rv_pad_rate.append(num_pads / batch_size)

        get_stats(src_lens, 'src', *stats_src)
        get_stats(trg_lens, 'trg', *stats_trg)

        if verbose:
            print('')
            for rv in stats_src: print(rv)
            print('')
            for rv in stats_trg: print(rv)
            print('')
            print('analyze_batch_idxs() finished.')
            print('')
        return stats_src, stats_trg

    def train_dataloader(self, batch_size, device):
        df = self.train_df.sample(frac=1).reset_index(drop=True)
        src_lens = df[self.src_tok.lang].map(len)
        trg_lens = df[self.trg_tok.lang].map(len)
        print('source max length:', max(src_lens), 'source min length:', min(src_lens))
        print('target max length:', max(trg_lens), 'target min length:', min(trg_lens))

        batch_groups, batch_samples = self.generate_groups(src_lens, trg_lens, batch_size)

        # bin packing
        df, batch_idxs = self.bin_packing(df, src_lens, trg_lens, batch_groups)

        stats_src, stats_trg = self.analyze_batch_idxs(batch_idxs, df, verbose=False)
        print('')
        for rv in stats_src:
            print(rv)
        print('')
        for rv in stats_trg:
            print(rv)
        print('')

        return batch_samples, torch.utils.data.DataLoader(
            df.values,
            batch_sampler=batch_idxs,
            collate_fn=collate_fn(device, self),
            pin_memory=True,
        )

    def valid_dataloader(self, batch_size, device):
        df = self.valid_df
        src_lens = df[self.src_tok.lang].map(len) + 2
        trg_lens = df[self.trg_tok.lang].map(len) + 2
        batch_groups, batch_samples = self.generate_groups(src_lens, trg_lens, batch_size)
        return batch_samples, torch.utils.data.DataLoader(
            df.values,
            batch_sampler=batch_groups,
            collate_fn=collate_fn(device, self),
            pin_memory=True,
        )

    def test_dataloader(self, batch_size, device):
        df = self.test_df
        src_lens = df[self.src_tok.lang].map(len) + 2
        trg_lens = df[self.trg_tok.lang].map(len) + 2
        batch_groups, batch_samples = self.generate_groups(src_lens, trg_lens, batch_size)

        return batch_samples, torch.utils.data.DataLoader(
            df.values,
            batch_sampler=batch_groups,
            collate_fn=collate_fn(device, self),
            pin_memory=True,
        )

    @staticmethod
    def pad(idss, idss_mask, tok, device='cpu'):
        """bos, eos, pad"""
        mlen = max(len(ids) for ids in idss)

        idss = [ids + [tok.pad_id] * (mlen - len(ids)) for ids in idss]
        idss_mask = [ids + [tok.pad_id] * (mlen - len(ids)) for ids in idss_mask]

        tensor = torch.tensor(idss, device=device).T
        tensor_mask = torch.tensor(idss_mask, device=device, dtype=torch.float).T
        return tensor, tensor_mask  # (l, b)

    @staticmethod
    def unpad(tensor, tok):  # (l, b)
        idss = tensor.T.tolist()
        idss = [l[1:] if l[0] == tok.bos_id else l for l in idss]
        idss = [l[:l.index(tok.eos_id)] if tok.eos_id in l else l for l in idss]
        return idss


class collate_fn:
    def __init__(self, device, dataset):
        self.device = device
        self.dataset = dataset

    def __call__(self, values):
        src_idss, trg_idss, src_mask, trg_mask = list(zip(*values))
        src, src_mask = self.dataset.pad(src_idss, src_mask, self.dataset.src_tok, self.device)
        trg, trg_mask = self.dataset.pad(trg_idss, trg_mask, self.dataset.trg_tok, self.device)
        return Batch(src, trg, src_mask, trg_mask)


class Batch:
    __slots__ = ['src_tensor', 'trg_tensor', 'src_mask', 'trg_mask']

    def __init__(self, src_tensor, trg_tensor, src_mask, trg_mask):
        self.src_tensor = src_tensor
        self.trg_tensor = trg_tensor
        self.src_mask = src_mask
        self.trg_mask = trg_mask


def packing1D(d,
              V_max,
              weight_pos=None,
              key=None,
              lower_bound=None,
              upper_bound=None,
              ):
    def get(lst, ndx):
        return [lst[n] for n in ndx]

    def argmax(lst):
        return max(range(len(lst)), key=lst.__getitem__)

    def revargsort(lst):
        return sorted(range(len(lst)), key=lambda i: -lst[i])

    isdict = isinstance(d, dict)

    if not hasattr(d, '__len__'):
        raise TypeError("d must be iterable")

    if not isdict and hasattr(d[0], '__len__'):
        if weight_pos is not None:
            key = lambda x: x[weight_pos]
        if key is None:
            raise ValueError("Must provide weight_pos or key for tuple list")

    if not isdict and key:
        new_dict = {i: val for i, val in enumerate(d)}
        d = {i: key(val) for i, val in enumerate(d)}
        isdict = True
        is_tuple_list = True
    else:
        is_tuple_list = False

    if isdict:

        # get keys and values (weights)
        keys_vals = d.items()
        keys = [k for k, v in keys_vals]
        vals = [v for k, v in keys_vals]

        # sort weights decreasingly
        ndcs = revargsort(vals)

        weights = get(vals, ndcs)
        keys = get(keys, ndcs)

        bins = [{}]
    else:
        weights = sorted(d, key=lambda x: -x)
        bins = [[]]

    # find the valid indices
    if lower_bound is not None and upper_bound is not None and lower_bound < upper_bound:
        valid_ndcs = filter(lambda i: lower_bound < weights[i] < upper_bound, range(len(weights)))
    elif lower_bound is not None:
        valid_ndcs = filter(lambda i: lower_bound < weights[i], range(len(weights)))
    elif upper_bound is not None:
        valid_ndcs = filter(lambda i: weights[i] < upper_bound, range(len(weights)))
    elif lower_bound is None and upper_bound is None:
        valid_ndcs = range(len(weights))
    elif lower_bound >= upper_bound:
        raise Exception("lower_bound is greater or equal to upper_bound")

    valid_ndcs = list(valid_ndcs)

    weights = get(weights, valid_ndcs)

    if isdict:
        keys = get(keys, valid_ndcs)

    # the total volume is the sum of all weights
    V_total = sum(weights)

    # prepare array containing the current weight of the bins
    weight_sum = [0.]

    # iterate through the weight list, starting with heaviest
    for item, weight in enumerate(weights):

        if isdict:
            key = keys[item]

        # find candidate bins where the weight might fit
        candidate_bins = list(filter(lambda i: weight_sum[i] + weight <= V_max, range(len(weight_sum))))

        # if there are candidates where it fits
        if len(candidate_bins) > 0:

            # find the fullest bin where this item fits and assign it
            candidate_index = argmax(get(weight_sum, candidate_bins))
            b = candidate_bins[candidate_index]

        # if this weight doesn't fit in any existent bin
        elif item > 0:
            # note! if this is the very first item then there is already an
            # empty bin open so we don't need to open another one.

            # open a new bin
            b = len(weight_sum)
            weight_sum.append(0.)
            if isdict:
                bins.append({})
            else:
                bins.append([])

        # if we are at the very first item, use the empty bin already open
        else:
            b = 0

        # put it in
        if isdict:
            bins[b][key] = weight
        else:
            bins[b].append(weight)

        # increase weight sum of the bin and continue with
        # next item
        weight_sum[b] += weight

    if not is_tuple_list:
        return bins
    else:
        new_bins = []
        for b in range(len(bins)):
            new_bins.append([])
            for _key in bins[b]:
                new_bins[b].append(new_dict[_key])
        return new_bins


def packing2D(d,
              V_max,
              weight_pos=None,
              key=None,
              lower_bound=None,
              upper_bound=None,
              ):
    def get(lst, ndx):
        return [lst[n] for n in ndx]

    def argmax(lst):
        return max(range(len(lst)), key=lst.__getitem__)

    def revargsort(lst):

        return sorted(range(len(lst)), key=lambda i: (-lst[i][0], -lst[i][1]))

    isdict = isinstance(d, dict)

    if not hasattr(d, '__len__'):
        raise TypeError("d must be iterable")

    if not isdict and hasattr(d[0], '__len__'):
        if weight_pos is not None:
            key = lambda x: x[weight_pos]
        if key is None:
            raise ValueError("Must provide weight_pos or key for tuple list")

    if not isdict and key:
        new_dict = {i: val for i, val in enumerate(d)}
        d = {i: key(val) for i, val in enumerate(d)}
        isdict = True
        is_tuple_list = True
    else:
        is_tuple_list = False

    if isdict:

        # get keys and values (weights)
        keys_vals = d.items()
        keys = [k for k, v in keys_vals]
        vals = [v for k, v in keys_vals]

        # sort weights decreasingly
        ndcs = revargsort(vals)

        weights = get(vals, ndcs)
        keys = get(keys, ndcs)

        bins = [{}]
    else:
        weights = sorted(d, key=lambda x: -x)
        bins = [[]]

    # find the valid indices
    if lower_bound is not None and upper_bound is not None and lower_bound < upper_bound:
        valid_ndcs = filter(lambda i: lower_bound < weights[i] < upper_bound, range(len(weights)))
    elif lower_bound is not None:
        valid_ndcs = filter(lambda i: lower_bound < weights[i], range(len(weights)))
    elif upper_bound is not None:
        valid_ndcs = filter(lambda i: weights[i] < upper_bound, range(len(weights)))
    elif lower_bound is None and upper_bound is None:
        valid_ndcs = range(len(weights))
    elif lower_bound >= upper_bound:
        raise Exception("lower_bound is greater or equal to upper_bound")

    valid_ndcs = list(valid_ndcs)

    weights = get(weights, valid_ndcs)

    if isdict:
        keys = get(keys, valid_ndcs)

    # the total volume is the sum of all weights
    V_total = (sum([w[0] for w in weights]), sum([w[1] for w in weights]))

    # prepare array containing the current weight of the bins
    weight_sum = [(0., 0.)]

    # iterate through the weight list, starting with heaviest
    for item, weight in enumerate(weights):

        if isdict:
            key = keys[item]

        # find candidate bins where the weight might fit
        candidate_bins = list(
            filter(lambda i: weight_sum[i][0] + weight[0] <= V_max[0] and weight_sum[i][1] + weight[1] <= V_max[1],
                   range(len(weight_sum))))

        # if there are candidates where it fits
        if len(candidate_bins) > 0:

            # find the fullest bin where this item fits and assign it
            candidate_index = argmax(get(weight_sum, candidate_bins))
            b = candidate_bins[candidate_index]

        # if this weight doesn't fit in any existent bin
        elif item > 0:
            # note! if this is the very first item then there is already an
            # empty bin open so we don't need to open another one.

            # open a new bin
            b = len(weight_sum)
            weight_sum.append((0., 0.))
            if isdict:
                bins.append({})
            else:
                bins.append([])

        # if we are at the very first item, use the empty bin already open
        else:
            b = 0

        # put it in
        if isdict:
            bins[b][key] = weight
        else:
            bins[b].append(weight)

        # increase weight sum of the bin and continue with
        # next item
        weight_sum[b] = (weight_sum[b][0] + weight[0], weight_sum[b][1] + weight[1])

    if not is_tuple_list:
        return bins
    else:
        new_bins = []
        for b in range(len(bins)):
            new_bins.append([])
            for _key in bins[b]:
                new_bins[b].append(new_dict[_key])
        return new_bins
