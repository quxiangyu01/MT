import math
import sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import mpl_axes_aligner


def get_result(file_path, metric):
    with open(file_path, "r", encoding="utf-8") as reader:
        delimiters = "=", "(", ")", ",", "\t", ":", " "
        regexPattern = '|'.join(map(re.escape, delimiters))
        lprob_e, perf = None, None
        for line in reader:
            line = line.strip()
            if "lprob" and "mean=" in line:
                line_parts = line.split()
                assert lprob_e is None
                lprob_e = float(line_parts[1])
            if metric in line:
                line_parts = line.split()
                assert perf is None
                perf = float(line_parts[1])
        if lprob_e is None: print(f"[error]: cannot extract lprob (base 2) from {file_path}")
        if perf is None: print(f"[error]: cannot extract {metric} from {file_path}")
        return lprob_e, perf

def generate_result_df(dir_path, metric):
    file_names = os.listdir(dir_path)
    result_df = pd.DataFrame(columns=["setting", "sorting_key", metric, "lprob_e", "lprob_10"])
    def make_sorting_key(s):
        order = {'sample': -1, 'vbs': 1, 'empty': 10000, 'ref': 20000}
        x = order[s.split(sep='_')[0]] * (float(s.split(sep='_')[1]) if len(s.split(sep='_')) > 1 else 1)
        # print(x)
        return x

    for file_name in file_names:
        if not file_name.endswith(".out"):
            continue
        setting = file_name[: file_name.index(".out")]
        file_path = os.path.join(dir_path, file_name)
        lprob_e, perf = get_result(file_path, metric)
        result_df.loc[result_df.shape[0]] = [setting, make_sorting_key(setting), perf, lprob_e, lprob_e/math.log(10)]

    # sort the table based on the setting name
    result_df = result_df.sort_values(by="sorting_key").drop('sorting_key', axis=1)
    return result_df





if __name__ == '__main__':
    # input_folder, output_file, metric = sys.argv[1], sys.argv[2], sys.argv[3]

    input_folder = "./" + sys.argv[1] if len(sys.argv) > 1 else "logP"
    output_file_prefix = input_folder+"/"+os.path.basename(input_folder)
    settings_to_print = {
        'empty':        r'empty output',
        'ref':          r'expected output',
        'sample_1.0':   r'sample ($\beta=1.0$)',
        'sample_0.75':  r'sample ($\beta=0.75$)',
        'sample_0.5':   r'sample ($\beta=0.5$)',
        'sample_0.25':  r'sample ($\beta=0.25$)',
        'vbs_1':        r'greedy',
        'vbs_4':        r'$|$beam$|=4$',
        'vbs_16':       r'$|$beam$|=16$',
        'vbs_64':       r'$|$beam$|=64$',
        'vbs_256':      r'$|$beam$|=256$',
        'vbs_1024':     r'$|$beam$|=1024$',
    }
    metric, metric_name = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else "cbleu_sAT", "BLEU"  # "cbleu_s ", "BLEU"
    ylims = {metric_name:(-2,50), 'lprob':(-70,0)}
    perf_of_ref = 42.4


    # get data
    result_df = generate_result_df(input_folder, metric)
    result_df.to_csv(output_file_prefix+'.csv', index=False)
    # print(result_df)


    # plot data
    result_df = result_df.loc[result_df['setting'].isin(settings_to_print)]
    result_df.loc[result_df['setting'] == 'ref', metric] = perf_of_ref
    print(result_df)

    settings = result_df['setting'].tolist()
    settings = [settings_to_print[s] for s in settings]
    perfs = result_df[metric].tolist()
    lprobs = result_df['lprob_10'].tolist()

    perf_color = 'C' + '0'
    lprob_color = 'C' + '1'
    perf_label = r'\textbf{' + metric_name + ' score (0-100)}'
    lprob_label = r'\textbf{avg.}~~$\log_{10}~ \mathbf{P}(\mathbf{y}|\mathbf{x}; \mathbf{w})$'
    legend_labels = (metric_name, r'$\log_{10}~ \mathbf{P}(\mathbf{w})$')

    matplotlib.rcParams.update({'font.size': 13})
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots()  #(figsize=(13/2, 10/2), dpi=300)
    ax.grid()
    ax2 = ax.twinx()

    # ax.plot(result_df["setting"], result_df["cblue_s"], marker="s", color="#e5323e", lw=4, label="BLEU", markersize=12)
    line_bleu = ax.plot(settings, perfs, marker="s", color=perf_color, label=perf_label, markersize=6)
    ax.set_ylabel(perf_label, color=perf_color, fontsize=15)
    ax.tick_params(axis='y', colors=perf_color)

    # ax2.plot(settings, result_df[lprob_col], marker="o", color="#003366", lw=4, label='log_10 P(w)', markersize=8)
    line_prob = ax2.plot(settings, lprobs, marker="o", color=lprob_color, label=lprob_label, markersize=4)
    ax2.set_ylabel(lprob_label, color=lprob_color, fontsize=15)
    ax2.tick_params(axis='y', colors=lprob_color)

    plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right', rotation_mode="anchor")

    plt.legend(line_bleu + line_prob, legend_labels, loc='upper left')
    # plt.legend(["cblue_s", lprob_label], loc="upper center", ncol=2)

    if ylims.get(metric_name) is not None:
        ax.set_ylim(*ylims[metric_name])
    if ylims.get('lprob') is not None:
        ax2.set_ylim(*ylims['lprob'])

    perf_greedy = perfs[4]
    lprob_greedy = lprobs[4]
    print(f'\n{metric_name}_greedy {perf_greedy}\tlprob_greedy {lprob_greedy}')
    mpl_axes_aligner.align.yaxes(ax, perf_greedy, ax2, lprob_greedy, pos=0.5)
    # ax2.set_yscale('log', basey=10)

    # pos1 = ax.get_position()  # get the original position
    # ax.spines['bottom'].set_position('zero')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    plt.tight_layout()
    print('save figure to ' + f'{output_file_prefix}.png')
    fig.savefig(f'{output_file_prefix}.png')


