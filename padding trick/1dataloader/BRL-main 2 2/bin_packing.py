from __future__ import print_function


def packing1D(d,
              V_max,
              weight_pos=None,
              key=None,
              lower_bound=None,
              upper_bound=None,
              ):
    """
    Distributes a list of weights, a dictionary of weights or a list of tuples containing weights
    to a minimal number of bins that have a fixed volume.

    Parameters
    ==========
    d : iterable
        list containing weights,
        OR dictionary where each (key,value)-pair carries the weight as value,
        OR list of tuples where one entry in the tuple is the weight. The position of
        this weight has to be given in optional variable weight_pos
    V_max : int or float
        Fixed bin volume
    weight_pos : int, default = None
        if d is a list of tuples, this integer number gives the position of the weight in a tuple
    key : function, default = None
        if d is a list, this key functions grabs the weight for an item
    lower_bound : float, default = None
        weights under this bound are not considered
    upper_bound : float, default = None
        weights exceeding this bound are not considered

    Returns
    =======
    bins : list
        A list. Each entry is a list of items or
        a dict of items, depending on the type of ``d``.
    """

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
    """
    Distributes a list of weights, a dictionary of weights or a list of tuples containing weights
    to a minimal number of bins that have a fixed volume.

    Parameters
    ==========
    d : iterable
        list containing weights,
        OR dictionary where each (key,value)-pair carries the weight as value,
        OR list of tuples where one entry in the tuple is the weight. The position of
        this weight has to be given in optional variable weight_pos
    V_max : int or float
        Fixed bin volume
    weight_pos : int, default = None
        if d is a list of tuples, this integer number gives the position of the weight in a tuple
    key : function, default = None
        if d is a list, this key functions grabs the weight for an item
    lower_bound : float, default = None
        weights under this bound are not considered
    upper_bound : float, default = None
        weights exceeding this bound are not considered

    Returns
    =======
    bins : list
        A list. Each entry is a list of items or
        a dict of items, depending on the type of ``d``.
    """

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


def packing(token_packed, group, src_lens, trg_lens):

    src_group_lens = src_lens[group]
    trg_group_lens = trg_lens[group]
    token_max_lens = (src_group_lens.max(), trg_group_lens.max())
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


def packing_test(token_packed, group, src_lens, trg_lens):
    token_max_lens = (max(src_lens), max(trg_lens))
    if token_packed == 'source-target':
        token_lens_dict = {i: (s, t) for i, s, t in zip(group, src_lens, trg_lens)}
        packs = packing2D(token_lens_dict, token_max_lens)
    elif token_packed == 'source':
        token_lens_dict = {i: s for i, s in zip(group, src_lens)}
        packs = packing1D(token_lens_dict, token_max_lens[0])
    elif token_packed == 'target':
        token_lens_dict = {i: t for i, t in zip(group, trg_lens)}
        packs = packing1D(token_lens_dict, token_max_lens[1])
    elif token_packed == 'max':
        token_lens_dict = {i: max(s, t) for i, s, t in zip(group, src_lens, trg_lens)}
        packs = packing1D(token_lens_dict, max(token_max_lens))

    return packs


source = {'1': 20, '2': 21, '3': 33, '4': 51, '5': 8, '6': 37, '7': 40, '8': 61, '9': 12, '10': 15}
target = {'1': 19, '2': 25, '3': 30, '4': 47, '5': 9, '6': 30, '7': 27, '8': 44, '9': 17, '10': 22}
group = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
token_packed = 'max'

res = packing_test(token_packed, group, list(source.values()), list(target.values()))
