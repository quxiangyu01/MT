from brl.utils import *

log_interval = int(sys.argv[1]) if len(sys.argv) >= 3 else 5000
num_logs = int(sys.argv[2]) if len(sys.argv) >= 3 else 20
bs_list = (1,)

def get_performance(beam_size):
    t, cbleu_t, rouge_t = [], [], []

    for iteration in interval(1, num_logs):
        step = iteration * log_interval
        fname = str(step) + '_bs' + str(beam_size) + '.out'
        try:
            data = open(fname).read().splitlines()
            print('reading '+fname+'.')
            
            # for search.py
            #cbleu = float(data[-3].split()[1])
            #sbleu = float(data[-4].split()[2])
            
            # for mle_model_analysis.py
            #cbleu = float(data[-5].split()[3])
            #sbleu = float(data[-6].split()[5])

            # for probability_model_analysis.py
            cbleu = float(data[-20].split()[1])
            rouge = float(data[-17].split()[1]) * 100
            
            t.append(step)
            cbleu_t.append(cbleu)
            rouge_t.append(rouge)
        except:
            pass

    return t, cbleu_t, rouge_t

perf_data = {}
for bs in bs_list:
    steps, cbleus, rouges = get_performance(beam_size=bs)
    perf_data[bs] = {'step': steps, 'cbleu': cbleus, 'rouge': rouges}
    print('step\tcbleu\trouge')
    for x, y, z in zip(steps, cbleus, rouges): print(f'{x}\t{y}\t{z}')
    print('')

plt.rcParams["figure.figsize"] = (10,8)
plot = LinePlot('# of gradient updates (x1000)', 'performance')
for bs in perf_data:
    steps, cbleus, rouges = perf_data[bs]['step'], perf_data[bs]['cbleu'], perf_data[bs]['rouge']
    plot.add_line(np.array([0]+steps)/1000, [0]+cbleus, label=f'corpus bleu (bs = {bs})')
    plot.add_line(np.array([0]+steps)/1000, [0]+rouges, label=f' avg. rouge (bs = {bs})')

plot.output('learning_curve.png')
