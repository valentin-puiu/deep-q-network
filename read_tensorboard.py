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
    
    event_acc.Reload()

    print(event_acc.Tags())

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
    #conv4 = event_acc.Histograms('mainDQN/conv4_1')


    steps = len(evaluation_score)
    x = np.arange(steps)
    y = np.zeros([steps, 1])

    for i in range(steps):
        y[i, 0] = evaluation_score[i][2] # value

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
    steps = len(loss)
    x = np.arange(steps)
    y = np.zeros([steps, 1])

    for i in range(steps):
        y[i, 0] = loss[i][2] # value

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

    steps = len(reward)
    x = np.arange(steps)
    y = np.zeros([steps, 1])

    for i in range(steps):
        y[i, 0] = reward[i][2] # value

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
    log_file = "/home/valentinpuiu/projects/deep-q-network/results/breakout/events.out.tfevents.1619207732.pop-os"
    plot_tensorflow_log(log_file)