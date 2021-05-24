import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import tensorboard
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path)
    event_acc_old = EventAccumulator('/home/valentinpuiu/projects/deep-q-network/results/enduro/events.out.tfevents.1621232947.pop-os')
    event_acc_new = EventAccumulator('/home/valentinpuiu/projects/deep-q-network/results/enduro/events.out.tfevents.1621307977.pop-os')
    event_acc_new2 = EventAccumulator('/home/valentinpuiu/projects/deep-q-network/results/enduro/events.out.tfevents.1621310059.pop-os')
    event_acc_old.Reload()
    event_acc_new.Reload()
    event_acc_new2.Reload()
    
    event_acc.Reload()

    print(event_acc.Tags())

    # Show all tags in the log file
    #print(event_acc.Tags())
    evaluation_score =   event_acc.Scalars('Performance/evaluation_score')
    loss = event_acc.Scalars('Performance/loss')
    reward = event_acc.Scalars('Performance/reward')
    conv1 = event_acc.Histograms('mainDQN/conv1_1')
    conv2 = event_acc.Histograms('mainDQN/conv2_1')
    conv3 = event_acc.Histograms('mainDQN/conv3_1')
    advantage = event_acc.Histograms('mainDQN/denseAdvantage')
    advantage_bias = event_acc.Histograms('mainDQN/denseAdvantageBias')
    value = event_acc.Histograms('mainDQN/denseValue')
    value_bias = event_acc.Histograms('mainDQN/denseValueBias')
    flat = event_acc.Histograms('mainDQN/flat')

    evaluation_score_old = event_acc_old.Scalars('Performance/evaluation_score')
    evaluation_score_new = event_acc_new.Scalars('Performance/evaluation_score')
    evaluation_score_new2 = event_acc_new2.Scalars('Performance/evaluation_score')
    loss_old = event_acc_old.Scalars('Performance/loss')
    loss_new = event_acc_new.Scalars('Performance/loss')
    loss_new2 = event_acc_new2.Scalars('Performance/loss')
    reward_old = event_acc_old.Scalars('Performance/reward')
    reward_new = event_acc_new.Scalars('Performance/reward')
    reward_new2 = event_acc_new2.Scalars('Performance/reward')
    steps = len(evaluation_score)
    
    old_steps = len(evaluation_score_old)
    new_steps = len(evaluation_score_new)
    new_steps2 = len(evaluation_score_new2)
    x = np.arange(old_steps + new_steps + new_steps2)
    y = np.zeros([old_steps + new_steps + new_steps2, 1])
    for i in range(old_steps):
        y[i, 0] = evaluation_score_old[i][2] # value

    for i in range(new_steps):
        y[old_steps + i, 0] = evaluation_score_new[i][2]

    for i in range(new_steps2):
        y[old_steps + new_steps + i, 0] = evaluation_score_new2[i][2]

    xnew = np.linspace(0,100,1000)
    y1 = np.nan_to_num(y, nan=0.0)
#    spl = make_interp_spline(x, y1, k=3)
#    y1 = spl(xnew)
    print(len(x))
    print(len(y1[:,0]))
    z = np.polyfit(x, y1[:,0], 1)
    p = np.poly1d(z)

    fig, ax = plt.subplots()
    ax.plot(x*200000, y, label='Scor la evaluare', alpha=1)
    ax.plot(x*200000, p(x), 'r--', label='Trend')
    ax.set_xlabel("Numar de pasi")
    ax.set_ylabel("Scor")
    plt.title("Scor la evaluare")
    ax.legend(loc='upper left', frameon=True)
    xlabels = ['{:,.2f}'.format(x) + 'M' for x in ax.get_xticks()/1000000]
    ax.set_xticklabels(xlabels)
    plt.show()

    # loss
    old_steps = len(loss_old)
    new_steps = len(loss_new)
    new_steps2 = len(loss_new2)
    x = np.arange(old_steps + new_steps + new_steps2)
    y = np.zeros([old_steps + new_steps + new_steps2, 1])
    for i in range(old_steps):
        y[i, 0] = loss_old[i][2] # value

    for i in range(new_steps):
        y[old_steps + i, 0] = loss_new[i][2]

    for i in range(new_steps2):
        y[old_steps + new_steps + i, 0] = loss_new2[i][2]

    xnew = np.linspace(0,100,1000)
    y1 = np.nan_to_num(y, nan=0.0)
#    spl = make_interp_spline(x, y1, k=3)
#    y1 = spl(xnew)
    print(len(x))
    print(len(y1[:,0]))
    z = np.polyfit(x, y1[:,0], 1)
    p = np.poly1d(z)

    fig, ax = plt.subplots()
    ax.plot(x*200000, y, label='Pierdere', alpha=1)
    ax.plot(x*200000, p(x), 'r--', label='Trend')
    ax.set_xlabel("Numar de pasi")
    ax.set_ylabel("Pierdere")
    plt.title("Pierdere")
    ax.legend(loc='upper left', frameon=True)
#    xlabels = ['{:,.2f}'.format(x) + 'M' for x in ax.get_xticks()/1000000]
    ax.set_xticklabels(xlabels)
    plt.show()

    ## recompensa

    old_steps = len(reward_old)
    new_steps = len(reward_new)
    new_steps2 = len(reward_new2)
    x = np.arange(old_steps + new_steps + new_steps2)
    y = np.zeros([old_steps + new_steps + new_steps2, 1])
    for i in range(old_steps):
        y[i, 0] = reward_old[i][2] # value

    for i in range(new_steps):
        y[old_steps + i, 0] = reward_new[i][2]

    for i in range(new_steps2):
        y[old_steps + i + new_steps , 0] = reward_new2[i][2] # value

    xnew = np.linspace(0,100,1000)
    y1 = np.nan_to_num(y, nan=0.0)
#    spl = make_interp_spline(x, y1, k=3)
#    y1 = spl(xnew)
    print(len(x))
    print(len(y1[:,0]))
    z = np.polyfit(x, y1[:,0], 1)
    p = np.poly1d(z)

    fig, ax = plt.subplots()
    ax.plot(x*200000, y, label='Recompensa', alpha=1)
    ax.plot(x*200000, p(x), 'r--', label='Trend')
    ax.set_xlabel("Numar de pasi")
    ax.set_ylabel("Recompensa")
    plt.title("Recompensa")
    ax.legend(loc='upper left', frameon=True)
#    xlabels = ['{:,.2f}'.format(x) + 'M' for x in ax.get_xticks()/1000000]
    ax.set_xticklabels(xlabels)
    plt.show()

    print(len(conv1[0].histogram_value.bucket_limit))


if __name__ == '__main__':
    log_file = "/home/valentinpuiu/projects/deep-q-network/results/breakout-gelu/events.out.tfevents.1620920971.pop-os"
    plot_tensorflow_log(log_file)