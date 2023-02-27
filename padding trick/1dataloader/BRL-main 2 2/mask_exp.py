import torch

# a = torch.tensor([[[2], [-4], [-4], [3], [2], [-5], [-5], [-5], [3], [0], [0]]])
# b = torch.bmm(a, a.transpose(1, 2))
# zero_attn_mask = b.eq(0)  # mask zero-padding
# sent_attn_mask = b.eq(20)  # mask sent-padding


# mask = sent_attn_mask + zero_attn_mask
#
# print(b[0])

def triu_mask(y_mask, num_heads=8):
    """x: (xlen, ...)"""

    q_mask, k_mask = y_mask.T.unsqueeze(-1), y_mask.T.unsqueeze(1)
    triu_mat = torch.bmm(q_mask, k_mask)  # bxsqxsk

    zero_attn_mask = triu_mat.eq(0)  # mask zero-padding

    sent_attn_mask = triu_mat.eq(20)  # mask sent-padding
    k_vmask = zero_attn_mask + sent_attn_mask
    triu_attn_mask = torch.triu(torch.ones(triu_mat.size()), 1).bool()

    mask = zero_attn_mask + sent_attn_mask + triu_attn_mask  # + triu_attn_mask_4 + triu_attn_mask_9

    mask = torch.cat([mask] * num_heads, 0)

    return mask  # (xlen, xlen)


a = torch.tensor([[-4], [-4], [-4], [-4], [-4], [-5], [-5], [-5], [-5], [0]])

mask = triu_mask(a)
#
from collections import Counter

# list = [1, -2, 1, -2, -2, -3, 4, 5, 4]
# result = dict(Counter(list))
# sen_len = [(k, result[k]) for k in sorted(result.keys())]
# print(sen_len)

import binpacking
b={'a':10,'b':10,'c':11,'d':1,'e':2,'f':7}
bins=binpacking.to_constant_bin_number(b,4)
print("===== dict\n",b,"\n",bins)
b=list(b.values())
bins=binpacking.to_constant_volume(b,11)
print("===== list\n",b,"\n",bins)



def packing_test(token_packed, group, src_lens, trg_lens):
    if token_packed == 'source-target':
        token_lens_dict = {i: (s, t) for i, s, t in zip(group, src_group_lens, trg_group_lens)}
        packs = packing2D(token_lens_dict, token_max_lens)
    elif token_packed == 'source':
        token_lens_dict = {i: s for i, s in zip(group, src_group_lens)}
        packs = packing1D(token_lens_dict, token_max_lens[0])
    elif token_packed == 'target':
        token_lens_dict = {i: t for i, t in zip(group, trg_group_lens)}
        packs = packing1D(token_lens_dict, token_max_lens[1])
    elif token_packed == 'max':
        token_lens_dict = {i: max(s, t) for i, s, t in zip(group, src_group_lens, trg_group_lens)}
        packs = packing1D(token_lens_dict, max(token_max_lens))

    return packs


source = {'1': 20, '2': 21, '3': 33, '4': 51, '5': 8, '6': 37, '7': 40, '8': 61, '9': 12, '10': 15}
target = {'1': 19, '2': 25, '3': 30, '4': 47, '5': 9, '6': 30, '7': 27, '8': 44, '9': 17, '10': 22}
group = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
token_packed = 'source-target'

res = packing_test(token_packed, group, source, target)