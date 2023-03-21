import torch
from torch import nn
import numpy as np
import pandas as pd
import os
import datetime
from .tdnn import AudioTDNN
from .config import model_config, train_config, val_config, test_config, device
from .config import exp_config, save_config, setup_training_params
from src.utils import AverageMeter, calculate_stats, load_label, worse_k_bar_fig
from .dataset import AudioDataset
# use tensorboardX rather than torch.utils.tensorboard
from tensorboardX import SummaryWriter



def train(model, train_loader, optimizer, scheduler, loss_fn, val_loader, writer):
    torch.set_grad_enabled(True)


    global_step, epoch = 1, 1
    save_config['metric']['Hparam/val_recall'] = 0
    best_epoch = 1
    best_val_stats = None

    # record the average training loss
    loss_meter = AverageMeter()

    

    model = model.to(device)
    model.train()
 
    while epoch < exp_config['n_epochs'] + 1:
        model.train()
        for i, (model_input, labels) in enumerate(train_loader):
                
            B = model_input.size(0)
            model_input = model_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            audio_output = model(model_input)
            loss = loss_fn(audio_output, labels)

            if global_step <= exp_config['warmup_end'] and (global_step % exp_config['warmup_step'] == 0 or global_step == 1):
                optimizer.param_groups[0]['lr'] = global_step / exp_config['warmup_end'] * exp_config['lr']

            loss_meter.update(loss.item(), B)
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            writer.add_scalar('Loss/avg_train_step', loss_meter.avg, global_step)
            writer.add_scalar('Learning Rate/base_lr', optimizer.param_groups[0]['lr'], global_step)


            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), exp_config['clip_grad_norm'])
            optimizer.step()

            global_step += 1

        # validate
        stats = validate(model, val_loader, loss_fn, writer, epoch)

        if stats['macro_recall'] > save_config['metric']['Hparam/val_recall']:
            best_val_stats = stats
            best_epoch = epoch
            save_config['metric']['Hparam/val_recall'] = stats['macro_recall']


            torch.save(model.state_dict(), save_config['best_model_path'])
        
        # normal scheduler every epoch
        scheduler.step()

        writer.add_scalar('Loss/train_epoch', loss_meter.avg, epoch)

        loss_meter.reset()
        epoch += 1
    
    return best_val_stats, best_epoch
    


def validate(model, val_loader, loss_fn, writer, epoch=None, split='val'):
    loss_meter = AverageMeter()
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)

    # switch to evaluate mode
    model.eval()


    A_predictions = []
    A_targets = []
    with torch.no_grad():
        for i, (model_input, target) in enumerate(val_loader):
            model_input = model_input.to(device)
            target = target.to(device)

            # compute output
            audio_output = model(model_input)

            # compute the loss
            loss = loss_fn(audio_output, target)

            predictions = audio_output.to('cpu').detach().softmax(dim=1)
            A_predictions.append(predictions)
            A_targets.append(target.to('cpu'))
            loss_meter.update(loss.item(), model_input.size(0))

        predictions = torch.cat(A_predictions)
        targets = torch.cat(A_targets)
        stats = calculate_stats(predictions, targets)
        stats["loss"] = loss_meter.avg

        one_hot_targets = torch.zeros(targets.shape[0], 418)
        one_hot_targets.scatter_(1, targets.unsqueeze(1), 1)
        writer.add_pr_curve(split.capitalize(), one_hot_targets, predictions, epoch)

        writer.add_scalar(f'Loss/{split}_epoch', loss_meter.avg, epoch)
        writer.add_scalar(f'Metrics/{split}_accuracy', stats['macro_acc'], epoch)
        writer.add_scalar(f'Metrics/{split}_recall', stats['macro_recall'], epoch)
        writer.add_scalar(f'Metrics/{split}_precision', stats['macro_precision'], epoch)
        writer.add_scalar(f'Metrics/{split}_f1', stats['macro_f1'], epoch)
        writer.add_scalar(f'Metrics/{split}_AP', stats['macro_avg_precision'], epoch)

    return stats



def save_results(best_val_stats, test_stats, writer):
    # save test metrics
    save_config['metric']['Hparam/test_loss'] = test_stats['loss']
    save_config['metric']['Hparam/test_recall'] = test_stats['macro_recall']
    save_config['metric']['Hparam/test_precision'] = test_stats['macro_precision']
    save_config['metric']['Hparam/test_f1'] = test_stats['macro_f1']
    save_config['metric']['Hparam/test_accuracy'] = test_stats['macro_acc']
    save_config['metric']['Hparam/test_avg_precision'] = test_stats['macro_avg_precision']

    # save test confusion matrix
    label = load_label(test_config['label_file'])
    df = pd.DataFrame(test_stats['confusion_matrix'], index=label, columns=label)
    df.to_csv(os.path.join(save_config['log_dir'], 'test_confusion_matrix.csv'), index=True, header=True)

    # save test worst k bar chart
    dataset = AudioDataset(model_config, test_config)
    fig = worse_k_bar_fig(test_stats, label, save_config['worse_k'], dataset, 'test')
    writer.add_figure('Worst/Test_Recall', fig)

    # save best val metrics
    save_config['metric']['Hparam/val_loss'] = best_val_stats['loss']
    save_config['metric']['Hparam/val_precision'] = best_val_stats['macro_precision']
    save_config['metric']['Hparam/val_f1'] = best_val_stats['macro_f1']
    save_config['metric']['Hparam/val_accuracy'] = best_val_stats['macro_acc']

    # save val worst k bar chart
    dataset = AudioDataset(model_config, val_config)
    fig = worse_k_bar_fig(best_val_stats, label, save_config['worse_k'], dataset, 'val')
    writer.add_figure('Worst/Val_Recall', fig)
    
    writer.add_text('Info', f'{ datetime.datetime.now() } Finish experiment')
    writer.close()

    
    writer = SummaryWriter(save_config['hparam_log_dir'], flush_secs=30)
    writer.add_hparams(exp_config, save_config['metric'], name=save_config['hparam_session_name'])
    writer.close()



if __name__ == '__main__':
    os.makedirs(save_config['log_dir'], exist_ok=True)
    os.makedirs(save_config['hparam_log_dir'], exist_ok=True)
    writer = SummaryWriter(save_config['log_dir'], flush_secs=30)
    
    writer.add_text('Info', f'{ datetime.datetime.now() } Start to load train and val data')
    train_loader = torch.utils.data.DataLoader(
            AudioDataset(model_config, train_config),
            batch_size=train_config['batch_size'], shuffle=True, 
            num_workers=train_config['num_workers'], pin_memory=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
            AudioDataset(model_config, val_config),
            batch_size=val_config['batch_size'], shuffle=True, 
            num_workers=val_config['num_workers'], pin_memory=False)
    writer.add_text('Info', f'{ datetime.datetime.now() } Finish loading train and val data')


    model = AudioTDNN(model_config)

    optimizer, scheduler, loss_fn = setup_training_params(model, writer)

    writer.add_text('Info', f'{ datetime.datetime.now() } Start to train the model')
    best_val_stats, best_epoch = train(model, train_loader, optimizer, scheduler, loss_fn, val_loader, writer)

    writer.add_text('Info', f'{ datetime.datetime.now() } Start to evaluate the model')
    # evaluate on test set
    test_loader = torch.utils.data.DataLoader(
            AudioDataset(model_config, test_config),
            batch_size=test_config['batch_size'], shuffle=True,
            num_workers=test_config['num_workers'], pin_memory=False)

    sd = torch.load(save_config['best_model_path'])
    model.load_state_dict(sd, strict=False)

    test_stats = validate(model, test_loader, loss_fn, writer, epoch=best_epoch, split='test')

    save_results(best_val_stats, test_stats, writer)