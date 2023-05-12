import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.nn as nn

def cross_entropy(preds, targets, reduction = 'none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
def plot_losses(log_path, fg_path):
    '''plot the loss history'''
    with open(log_path, 'r') as f:
        lines = f.readlines()
        steps = [int(line.strip().split(',')[0].split(': ')[1]) for line in lines[1:]]
        losses = [round(float(line.strip().split(',')[1].split(': ')[1]), 3) for line in lines[1:]]
    plt.plot(steps, losses, label = fg_path.split('/')[-1].split('.')[0])
    plt.legend()
    plt.savefig(fg_path)
    # plt.close()

def plot_histo(log_path, fg_path):
    data = pd.read_csv(log_path)
    data.hist(bins = 50)
    plt.savefig(fg_path)

def to_device(data, device):
    type_list = [torch.int64, torch.int32, torch.float64, torch.float32]
    for k, v in data.items():
        if type(v) == torch.Tensor:
            if v.dtype in type_list:
                data[k] = data[k].to(device)
    return data

if __name__ == '__main__':
    src_path = 'outputs/logs/train_log'
    test_path = 'outputs/logs/test_log'
    plot_losses(f'{src_path}.txt', f'{src_path}.png')
    plot_losses(f'{test_path}.txt', f'{test_path}.png')
    # plot_histo(os.path.join(src_path, 'wav_len.csv'), os.path.join(src_path, 'histo.png'))


