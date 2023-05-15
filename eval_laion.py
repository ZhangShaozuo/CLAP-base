import argparse
import laion_clap
import librosa
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
    
    ax.scatter(x, y, c = points_color, s = 10, alpha = 0.75)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    ax.legend(handles = legend_elements, loc="upper right", title="Classes")
    plt.savefig(f'{save_path_head}/{title}.png')

def plot_tSNE(args, config):
    '''Load the tSNE of wav/text embedding and plot it'''
    save_path_head = os.path.join(config['save_path'], 'laion_clap')
    os.makedirs(os.path.join(save_path_head, 'tSNE'), exist_ok=True)
    with open(os.path.join(save_path_head, f'{args.type}_embeds.pkl'), 'rb') as f:
        embeds_d = pickle.load(f)
    # single_label = [{k:v.shape[0]} for k, v in embeds_d.items() if len(k.split(',')) == 1]
    # majority_label = {k:v.shape[0] for k, v in embeds_d.items() if v.shape[0]>40}
    majority_label = {k:v.shape[0] for k, v in embeds_d.items()} 
    label_list = ['sadness', 'neutral', 'joy', 'surprise', 'anger', 'fear', 'disgust']
    print(majority_label)
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', u'orchid', u'darkcyan', u'grey', u'dodgerblue', u'turquoise', u'darkviolet']
    colors = [mc for mc in mcolors.TABLEAU_COLORS.keys()]
    tmp, points_c, color_labels = [], [], {}
    
    count = 0
    for k in majority_label.keys():
        tmp.append(embeds_d[k])
        points_c.append([colors[count]]*embeds_d[k].shape[0])
        color_labels[colors[count]] = label_list[k]
        count += 1
    points_c = [item for sublist in points_c for item in sublist]
    embeds = torch.cat(tmp, dim=0).squeeze(1).cpu()
    for i in tqdm(range(5,55,5)):
        tsne = TSNE(perplexity=i, n_components=2, init = 'pca', n_iter=2000, random_state=2, early_exaggeration=10)
        x_fit_transform = tsne.fit_transform(embeds)
        plot_2d(x_fit_transform, points_color = points_c, color_labels = color_labels, title = f'{args.type}_perplex_{i}', save_path_head = os.path.join(save_path_head,'tSNE'))

def save_la_embeds(args, config):
    '''Calculate the tSNE of wav/text embedding and save it'''
    test_dataset = load_from_disk(os.path.join(config['data_path'], 'dev_dataset_for_laion'))
    label_list = ['sadness', 'neutral', 'joy', 'surprise', 'anger', 'fear', 'disgust']
    ckpt_path = os.path.join(config['save_path'], 'laion_clap/630k-audioset-best.pt')
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt(ckpt_path)
    model.eval()
    save_path = os.path.join(config['save_path'], 'laion_clap')
    os.makedirs(save_path, exist_ok=True)
    progress_bar = tqdm(range(len(test_dataset)))
    a_embeds, l_embeds = {}, {}
    for step, batch in enumerate(test_dataset):
        with torch.no_grad():
            audio_data, _ = librosa.load(batch['wav_path'], sr=48000)
            audio_data = audio_data.reshape(1, -1)
            audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False)
            audio_embed = torch.from_numpy(audio_embed)
            key = batch['labels']
            
            text_embed = model.get_text_embedding([batch[f'input_ids_{key}'], 'This is an auxiliary sent'])[0]
            text_embed = torch.from_numpy(text_embed)
            
            if key not in a_embeds.keys():
                a_embeds[key] = audio_embed.unsqueeze(0)
                l_embeds[key] = text_embed.unsqueeze(0)
            else:
                a_embeds[key] = torch.cat((a_embeds[key], audio_embed.unsqueeze(0)), dim=0) 
                l_embeds[key] = torch.cat((l_embeds[key], text_embed.unsqueeze(0)), dim=0) 
            progress_bar.update(1)
    with open(os.path.join(save_path, 'a_embeds.pkl'), 'wb') as f:
        pickle.dump(a_embeds, f)
    with open(os.path.join(save_path, 'l_embeds.pkl'), 'wb') as f:
        pickle.dump(l_embeds, f)

def eval_laion(args, config):
    '''Calculate the tSNE of wav/text embedding and save it'''
    test_dataset = load_from_disk(os.path.join(config['data_path'], 'dev_dataset_for_laion'))
    label_list = ['sadness', 'neutral', 'joy', 'surprise', 'anger', 'fear', 'disgust']
    ckpt_path = os.path.join(config['save_path'], 'laion_clap/630k-audioset-best.pt')
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt(ckpt_path)
    model.eval()
    save_path = os.path.join(config['save_path'], 'laion_clap')
    os.makedirs(save_path, exist_ok=True)
    progress_bar = tqdm(range(len(test_dataset)))
    y_preds, y_trues = [], []
    for step, batch in enumerate(test_dataset):
        with torch.no_grad():
            cos_sims = []
            audio_data, _ = librosa.load(batch['wav_path'], sr=48000)
            audio_data = audio_data.reshape(1, -1)
            audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False)
            audio_embed = torch.from_numpy(audio_embed)

            text_data = [batch[f'input_ids_{idx}'] for idx in range(len(label_list))]
            text_embed = model.get_text_embedding(text_data)
            text_embed = torch.from_numpy(text_embed)
            
            cos_sims = [(CosineSimilarity(dim=1)(audio_embed, text_embed[idx])) for idx in range(len(label_list))]
            cos_sims = torch.stack(cos_sims, dim=1)
            
            preds = torch.argmax(cos_sims, dim=1).cpu()
            y_preds.append(int(preds[0]))
            y_trues.append(batch['labels'])
            progress_bar.update(1)
    print(classification_report(y_trues, y_preds, target_names=label_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='a', required=False)
    args = parser.parse_args()

    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    # save_la_embeds(args, config)
    plot_tSNE(args, config)
    args.type = 'l'
    plot_tSNE(args, config)
    # eval_laion(args, config)