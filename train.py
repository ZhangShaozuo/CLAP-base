
import argparse
import os
import yaml

from dataset import DataCollatorCTCWithPadding
from datasets import load_from_disk
from model import CLAP
from tqdm.auto import tqdm
from transformers import Wav2Vec2Processor, BertTokenizer, get_scheduler
from utils import to_device
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import init_process_group
from torch.cuda.amp import GradScaler, autocast


def train(rank, config, num_gpus):
    device = torch.device('cuda:{:d}'.format(config['train']['device_ids'][rank]))
    print("----------------------\n Device used: {}\n----------------------".format(device))
    train_dataset= load_from_disk(os.path.join(config['data_path'], 'train_dataset'))
    test_dataset = load_from_disk(os.path.join(config['data_path'], 'dev_dataset'))
    # unused_features = ['wav_path', 'id', 'transcr', 'emotions', 'token_count', 'label']
    # train_dataset= load_from_disk(os.path.join(config['data_path'], 'train_dataset')).remove_columns(unused_features)
    # test_dataset = load_from_disk(os.path.join(config['data_path'], 'test_dataset')).remove_columns(unused_features)
    processor = Wav2Vec2Processor.from_pretrained(config['audio_model']['ckpt'])
    tokenizer = BertTokenizer.from_pretrained(config['language_model']['ckpt'])
    if num_gpus > 1:
        init_process_group(backend = config["dist_config"]['dist_backend'],
                           init_method = config["dist_config"]['dist_url'],
                           world_size = config['dist_config']['world_size'] * num_gpus, 
                           rank = rank)
        data_sampler = DistributedSampler(train_dataset)
    else:
        data_sampler = None
    # Note: It is suggested to map the dataset to cuda devices during training, not in data data_collator
    data_collator = DataCollatorCTCWithPadding(processor = processor,
                                               tokenizer = tokenizer, 
                                               padding = True)
    train_loader = DataLoader(train_dataset, 
                              batch_size = config['train']['batch_size'],
                              sampler = data_sampler,
                              shuffle = True,
                              collate_fn = data_collator)
    test_loader = DataLoader(test_dataset, 
                             batch_size = config['train']['batch_size'], 
                             shuffle=False, 
                             collate_fn=data_collator)
    log_step = config['train']['log_step']
    eval_step = config['train']['eval_step']
    save_step = config['train']['save_step']
    grad_accum = config['train']['gradient_accumulation']
    save_path = config['save_path']
    model = CLAP(config).to(device)
    scaler = GradScaler()
    if num_gpus > 1:
        model = DistributedDataParallel(model, 
                                        device_ids = [config['train']['device_ids'][rank]], 
                                        find_unused_parameters=True)
    optimizer = AdamW(model.parameters(), lr = config['train']['learning_rate'])
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=config['train']['total_step'])
    # lr_scheduler = StepLR(optimizer, step_size = config['train']['lr_decay_step'], gamma = config['train']['lr_decay_rate'])
    if rank == 0:
        num_param = sum(param.numel() for param in model.parameters())
        os.makedirs(os.path.join(config['save_path'], 'checkpoints'), exist_ok = True)
        os.makedirs(os.path.join(config['save_path'], 'logs'), exist_ok = True)
        os.makedirs(os.path.join(config['save_path'], 'results'), exist_ok = True)  
        with open(os.path.join(config['save_path'], 'logs', 'train_log.txt'), 'w') as f:
            f.write('The number of parameters of the model is {:d}.\n'.format(num_param))
        with open(os.path.join(config['save_path'], 'logs', 'test_log.txt'), 'w') as f:
            f.write('Start evaluation:\n')
        outer_bar = tqdm(range(config['train']['total_step']))

    train = True
    model.train()
    epoch, step = 1, 1
    loss_train, accum_step = 0, 0
    max_mem = 0
    while train:
        if rank == 0:
            inner_bar = tqdm(total=len(train_loader), desc="Epoch {}".format(epoch), position=1)
        if num_gpus > 1:
            data_sampler.set_epoch(epoch)
        for batch in train_loader:
            if train == False:
                break
            batch = to_device(batch, device)
            try:
                with autocast():
                    outputs = model(input_ids = batch['input_ids'], 
                                    l_attn_mask = batch['l_attn_mask'],
                                    input_values = batch['input_values'],
                                    a_attn_mask = batch['a_attn_mask'])
                    mem = torch.cuda.memory_allocated(device = device) / 1024**2
                    if mem>max_mem:
                        max_mem = mem
                    loss = outputs.loss
            except RuntimeError as e:
                print(e)
                print('Shape of wav inputs: ', batch['input_values'].shape)
                breakpoint()
            scaler.scale(loss / grad_accum).backward()
            if step % grad_accum == 0:
                # clip_grad_norm_(model.parameters(), 1)
                
                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            if rank == 0:
                loss_train += loss.item()
                accum_step += 1
                if step % log_step == 0:
                    mem = torch.cuda.memory_allocated(device = device) / 1024**2
                    message = f"Step: {step}, Training loss: {loss_train/(accum_step+1)}, Memory allocated/Cur max: {mem}/{max_mem}\n"
                    outer_bar.write(message, end='')
                    with open(os.path.join(config['save_path'], 'logs', 'train_log.txt'), 'a') as f:
                        f.write(message)
                    loss_train, accum_step = 0, 0
                if step % eval_step == 0 or step == 1:
                    model.eval()
                    eval_bar = tqdm(total=len(test_loader), desc="Evaluating", position=2)
                    with torch.no_grad():
                        loss_test = 0
                        # acc_h, acc_v = 0, 0
                        acc = 0
                        for batch in test_loader:
                            batch = to_device(batch, device)
                            outputs = model(input_ids = batch['input_ids'], 
                                            l_attn_mask = batch['l_attn_mask'],
                                            input_values = batch['input_values'],
                                            a_attn_mask = batch['a_attn_mask'])
                            loss, logits = outputs.loss, outputs.logits
                            logits_v, logits_h = torch.argmax(logits, dim = 0), torch.argmax(logits, dim = 1)
                            labels = torch.arange(batch['input_ids'].shape[0]).to(device)

                            acc_h = torch.sum(logits_h == labels)
                            acc_v = torch.sum(logits_v == labels)
                            acc = torch.sum(acc_h == acc_v)
                            loss_test += loss.item()
                            eval_bar.update(1)
                    message = f"Step: {step}, Testing loss: {loss_test/len(test_loader)}"
                    message_acc = f", Testing accuracy: {acc/len(test_dataset)}\n"
                    outer_bar.write(message + message_acc, end='')
                    with open(os.path.join(config['save_path'], 'logs', 'test_log.txt'), 'a') as f:
                        f.write(message + message_acc)
                    model.train()
                if step % save_step == 0:
                    os.makedirs(f'{save_path}/checkpoints', exist_ok = True)
                    torch.save({'model': model.state_dict(), 
                                'optimizer': optimizer.state_dict()}, 
                            os.path.join(save_path, 'checkpoints', f'CLAP-{step}.pth.tar'))
            if step == config['train']['total_step']:
                train = False
                break
            step += 1
            if rank == 0:
                outer_bar.update(1)
                inner_bar.update(1)
        epoch += 1            

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type = int, required=False)
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    num_gpus = len(config['train']['device_ids'])
    if num_gpus > 1:

        mp.spawn(train, nprocs = num_gpus, args = (config, num_gpus))
    else:
        train(0, config, num_gpus)
