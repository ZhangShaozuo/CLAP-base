import argparse
import os
import pickle
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dataset import DataCollatorPNWithPadding, DataCollatorCTCWithPadding
from datasets import load_from_disk
from matplotlib import ticker
from matplotlib.patches import Patch
from model import CLAP
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from transformers import Wav2Vec2Processor, BertTokenizer
from torch.nn import CosineSimilarity
from torch.utils.data import DataLoader
from utils import to_device

def plot_2d(points, points_color, color_labels, title, save_path_head):
    x, y = points.T
    legend_elements = [Patch(facecolor=color, label=color_labels[color]) for color in set(points_color)]

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    
    ax.scatter(x, y, c = points_color, s = 5, alpha = 0.75)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    ax.legend(handles = legend_elements, loc="upper right", title="Classes")
    plt.savefig(f'{save_path_head}/{title}.png')

def plot_tSNE(args, config):
    '''Load the tSNE of wav/text embedding and plot it'''
    save_path_head = os.path.join(config['save_path'], str(args.restore_step))
    os.makedirs(os.path.join(save_path_head, 'tSNE'), exist_ok=True)
    with open(os.path.join(save_path_head, f'{args.type}_embeds.pkl'), 'rb') as f:
        embeds_d = pickle.load(f)
    # single_label = [{k:v.shape[0]} for k, v in embeds_d.items() if len(k.split(',')) == 1]
    # majority_label = {k:v.shape[0] for k, v in embeds_d.items() if v.shape[0]>40}
    majority_label = {k:v.shape[0] for k, v in embeds_d.items()} 
    print(majority_label)
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', u'orchid', u'darkcyan', u'grey', u'dodgerblue', u'turquoise', u'darkviolet']
    colors = [mc for mc in mcolors.TABLEAU_COLORS.keys()]
    tmp, points_c, color_labels = [], [], {}
    
    count = 0
    for k in majority_label.keys():
        tmp.append(embeds_d[k])
        points_c.append([colors[count]]*embeds_d[k].shape[0])
        color_labels[colors[count]] = k
        count += 1
    points_c = [item for sublist in points_c for item in sublist]
    embeds = torch.cat(tmp, dim=0).cpu()
    for i in tqdm(range(5,55,5)):
        tsne = TSNE(perplexity=i, n_components=2, init = 'pca', n_iter=2000, random_state=2, early_exaggeration=10)
        x_fit_transform = tsne.fit_transform(embeds)
        plot_2d(x_fit_transform, points_color = points_c, color_labels = color_labels, title = f'{args.type}_perplex_{i}', save_path_head = os.path.join(save_path_head,'tSNE'))

def save_la_embeds(args, config):
    '''Calculate the tSNE of wav/text embedding and save it'''
    device = torch.device(f'cuda:{args.device}')
    test_dataset = load_from_disk(os.path.join(config['data_path'], 'dev_dataset'))
    processor = Wav2Vec2Processor.from_pretrained(config['audio_model']['ckpt'])
    tokenizer = BertTokenizer.from_pretrained(config['language_model']['ckpt'])
    data_collator = DataCollatorCTCWithPadding(processor = processor,tokenizer = tokenizer, padding = True, device=device)
    test_loader = DataLoader(test_dataset, batch_size=config['test']['batch_size'], shuffle=False, collate_fn=data_collator)

    model = CLAP(config).to(device)
    ckpt_path = os.path.join(config['save_path'], 'checkpoints', 'CLAP-{}.pth.tar'.format(args.restore_step))
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    save_path = os.path.join(config['save_path'], str(args.restore_step))
    os.makedirs(save_path, exist_ok=True)
    progress_bar = tqdm(range(len(test_loader)))
    embeds_d, l_embeds = {}, {}
    for step, batch in enumerate(test_loader):
        batch = to_device(batch, device)
        with torch.no_grad():
            outputs = model(input_ids = batch['input_ids'], 
                            l_attn_mask = batch['l_attn_mask'],
                            input_values = batch['input_values'],
                            a_attn_mask = batch['a_attn_mask'])
            for idx in range(batch['input_ids'].shape[0]):
                key = batch['labels'][idx]
                if key not in embeds_d.keys():
                    embeds_d[key] = outputs.a_last_hidden_states[idx].unsqueeze(0)
                    l_embeds[key] = outputs.l_last_hidden_states[idx].unsqueeze(0)
                else:
                    embeds_d[key] = torch.cat((embeds_d[key], outputs.a_last_hidden_states[idx].unsqueeze(0)), dim=0) 
                    l_embeds[key] = torch.cat((l_embeds[key], outputs.l_last_hidden_states[idx].unsqueeze(0)), dim=0) 
            progress_bar.update(1)
    with open(os.path.join(save_path, 'a_embeds.pkl'), 'wb') as f:
        pickle.dump(embeds_d, f)
    with open(os.path.join(save_path, 'l_embeds.pkl'), 'wb') as f:
        pickle.dump(l_embeds, f)

def eval_PN(args, config):
    '''Calculate the tSNE of wav/text embedding and save it'''
    device = torch.device(f'cuda:{args.device}')
    test_dataset = load_from_disk(os.path.join(config['data_path'], 'dev_PN_dataset'))
    label_list = ['sadness', 'neutral', 'joy', 'surprise', 'anger', 'fear', 'disgust']
    processor = Wav2Vec2Processor.from_pretrained(config['audio_model']['ckpt'])
    tokenizer = BertTokenizer.from_pretrained(config['language_model']['ckpt'])
    data_collator = DataCollatorPNWithPadding(processor = processor, tokenizer = tokenizer, num_labels = len(label_list), padding = True, device=device)
    test_loader = DataLoader(test_dataset, batch_size=config['test']['batch_size'], shuffle=False, collate_fn=data_collator)

    model = CLAP(config).to(device)
    ckpt_path = os.path.join(config['save_path'], 'checkpoints', 'CLAP-{}.pth.tar'.format(args.restore_step))
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    save_path = os.path.join(config['save_path'], str(args.restore_step))
    os.makedirs(save_path, exist_ok=True)
    progress_bar = tqdm(range(len(test_loader)))
    embeds_d, l_embeds = {}, {}
    y_preds, y_trues = [], []
    for step, batch in enumerate(test_loader):
        batch = to_device(batch, device)
        with torch.no_grad():
            cos_sims = []
            for idx in range(len(label_list)):
                outputs = model(input_ids = batch[f'input_ids_{idx}'], 
                                l_attn_mask = batch[f'l_attn_mask_{idx}'],
                                input_values = batch['input_values'],
                                a_attn_mask = batch['a_attn_mask'])
                cos_sims.append(CosineSimilarity(dim=1)(outputs.a_last_hidden_states, outputs.l_last_hidden_states))
            cos_sims = torch.stack(cos_sims, dim=1)
            preds = torch.argmax(cos_sims, dim=1).cpu()
            y_preds.append(preds)
            y_trues.append(batch['labels'])
            progress_bar.update(1)
    y_preds = [pred for sublist in y_preds for pred in sublist]
    y_trues = [true for sublist in y_trues for true in sublist]
    print(classification_report(y_trues, y_preds, target_names=label_list))

'''command
python3 tSNE.py --restore_step 20000 --device 1 --type a
python3 tSNE.py --restore_step 20000 --device 1 --type l
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=str, required=True)
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--type', type=str, default='a')
    args = parser.parse_args()

    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    # save_la_embeds(args, config)
    # plot_tSNE(args, config)
    eval_PN(args, config)