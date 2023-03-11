import torch
from torch import nn
import numpy as np
import pandas as pd
import os
import datetime
from .ast_models import ASTModel
from .config import model_config, train_config, val_config, test_config, device
from .config import exp_config, save_config, test_confusion_matrix_path
from src.utils import AverageMeter, calculate_stats, load_label
from .dataset import AudioDataset
from tensorboardX import SummaryWriter



def train(audio_model, train_loader, val_loader, writer):
    torch.set_grad_enabled(True)

    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)

    global_step, epoch = 0, 0
    save_config['metric']['Hparam/val_recall'], best_epoch = 0, 0

    # record the average training loss
    loss_meter = AverageMeter()

    optimizer, normal_scheduler = set_optimizer_scheduler(audio_model, writer)
    loss_fn = nn.BCEWithLogitsLoss()

    audio_model = audio_model.to(device)
    audio_model.train()
 
    while epoch < exp_config['n_epochs']:
        audio_model.train()
        for i, (audio_input, labels) in enumerate(train_loader):
                
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            audio_output = audio_model(audio_input, 'ft_cls')
            loss = loss_fn(audio_output, labels)

            if global_step <= exp_config['warmup_end'] and global_step % exp_config['warmup_step'] == 0:
                optimizer.param_groups[0]['lr'] = global_step / exp_config['warmup_end'] * exp_config['lr']
                optimizer.param_groups[1]['lr'] = global_step / exp_config['warmup_end'] * exp_config['lr'] * exp_config['head_lr']

            loss_meter.update(loss.item(), B)
            writer.add_scalar('Loss/train_avg', loss_meter.avg, global_step)
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Learning Rate/base', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Learning Rate/head_mlp', optimizer.param_groups[1]['lr'], global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # warmup
            # if global_step <= exp_config['warmup_end'] and global_step % exp_config['warmup_step'] == 0:
            #     warmup_scheduler.step()

            global_step += 1

        # validate
        stats = validate(audio_model, val_loader, writer, epoch)

        if stats['macro_recall'] > save_config['metric']['Hparam/val_recall']:
            save_config['metric']['Hparam/val_loss'] = stats['val_loss']
            save_config['metric']['Hparam/val_recall'] = stats['macro_recall']
            save_config['metric']['Hparam/val_precision'] = stats['macro_precision']
            save_config['metric']['Hparam/val_f1'] = stats['macro_f1']
            save_config['metric']['Hparam/val_accuracy'] = stats['macro_acc']

            best_epoch = epoch
            torch.save(audio_model.state_dict(), save_config['best_model_path'])
        
        # normal scheduler every epoch
        normal_scheduler.step()

        writer.add_scalar('Loss/train_avg_epoch', loss_meter.avg, epoch)
        epoch += 1
    

    


def validate(audio_model, val_loader, writer, epoch=0, split='val'):
    loss_meter = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)

    # switch to evaluate mode
    audio_model.eval()


    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, target) in enumerate(val_loader):
            audio_input = audio_input.to(device)
            target = target.to(device)

            # compute output
            audio_output = audio_model(audio_input, 'ft_cls')

            # compute the loss
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(audio_output, target)

            predictions = audio_output.to('cpu').detach().sigmoid()
            A_predictions.append(predictions)
            A_targets.append(target.to('cpu'))
            A_loss.append(loss.to('cpu').detach())
            loss_meter.update(loss.item(), audio_input.size(0))

        predictions = torch.cat(A_predictions)
        targets = torch.cat(A_targets)
        loss = loss_meter.avg
        stats = calculate_stats(predictions, targets)
        stats["val_loss"] = loss

        writer.add_pr_curve(split.capitalize(), targets, predictions, epoch)

        if split == 'val':
            writer.add_scalar(f'Loss/val_epoch', loss, epoch)
            writer.add_scalar('Val_Metrics/macro_accuracy', stats['macro_acc'], epoch)

            writer.add_scalar('Val_Metrics/macro_recall', stats['macro_recall'], epoch)

            writer.add_scalar('Val_Metrics/macro_precision', stats['macro_precision'], epoch)

            writer.add_scalar('Val_Metrics/macro_f1', stats['macro_f1'], epoch)

            writer.add_scalar('Val_Metrics/macro_AP', stats['macro_avg_precision'], epoch)

        elif split == 'test':
            save_config['metric']['Hparam/test_loss'] = loss
            save_config['metric']['Hparam/test_recall'] = stats['macro_recall']
            save_config['metric']['Hparam/test_precision'] = stats['macro_precision']
            save_config['metric']['Hparam/test_f1'] = stats['macro_f1']
            save_config['metric']['Hparam/test_accuracy'] = stats['macro_acc']
            save_config['metric']['Hparam/test_avg_precision'] = stats['macro_avg_precision']

            # save test confusion matrix
            label = load_label(test_config['label_file'])
            df = pd.DataFrame(stats['confusion_matrix'], index=label, columns=label)
            df.to_csv(test_confusion_matrix_path, index=True, header=True)

            # save top k worst save_config['metric'] class pr curve in test
            worse_k_index = np.argsort(stats['class_wise_recall'])[:save_config['worse_k']]
            for i, k in enumerate(worse_k_index):
                writer.add_pr_curve(f'Test_worse_K/Top_worse_{i}_{ label[k] }', targets[:, k], predictions[:, k])


    return stats


def set_optimizer_scheduler(audio_model, writer):
    
    # calculate statistics for the model
    save_config['metric']['Hparam/model_params'] = sum(p.numel() for p in audio_model.parameters()) / 1e6
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    writer.add_text('Model', 'Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))


    # diff lr optimizer for mlp head
    mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias']
    mlp_params = list(filter(lambda kv: kv[0] in mlp_list, audio_model.module.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in mlp_list, audio_model.module.named_parameters()))
    mlp_params = [i[1] for i in mlp_params]
    base_params = [i[1] for i in base_params]

    # only finetuning small/tiny models on balanced audioset uses different learning rate for mlp head
    optimizer = torch.optim.Adam([{'params': base_params, 'lr': exp_config['lr']}, 
                                  {'params': mlp_params, 'lr': exp_config['lr'] * exp_config['head_lr']}],
                                 weight_decay=exp_config['weight_decay'], betas=(0.95, 0.999))
    
    writer.add_text('Model', 'The mlp parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    writer.add_text('Model', 'The base parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))
    

    # set scheduler will reset the initial lr set by the optimizer to 0.
    # warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda step: (step / exp_config['warmup_end']) * exp_config['lr'], 
    #                                                                            lambda step: (step / exp_config['warmup_end']) * exp_config['lr'] * exp_config['head_lr']])



                                                                               
    normal_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(exp_config['lrscheduler_start'], exp_config['lrscheduler_end'], exp_config['lrscheduler_step'])),gamma=exp_config['lrscheduler_gamma'])

    return optimizer, normal_scheduler





if __name__ == '__main__':
    os.makedirs(save_config['log_dir'], exist_ok=True)
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


    audio_model = ASTModel(label_dim=model_config['num_classes'], fshape=model_config['fshape'],
                        tshape=model_config['tshape'], fstride=model_config['fstride'], tstride=model_config['tstride'],
                        input_fdim=model_config['num_mel_bins'], input_tdim=model_config['target_length'],
                        model_size=model_config['model_size'], pretrain_stage=False,
                        load_pretrained_mdl_path=model_config['pretrained_mdl_path'])

    writer.add_text('Info', f'{ datetime.datetime.now() } Start to train the model')
    train(audio_model, train_loader, val_loader, writer)

    writer.add_text('Info', f'{ datetime.datetime.now() } Start to evaluate the model')
    # evaluate on test set
    test_loader = torch.utils.data.DataLoader(
            AudioDataset(model_config, test_config),
            batch_size=test_config['batch_size'], shuffle=True,
            num_workers=test_config['num_workers'], pin_memory=False)

    sd = torch.load(save_config['best_model_path'])
    audio_model.load_state_dict(sd, strict=False)

    validate(audio_model, test_loader, writer, epoch=exp_config['n_epochs'] - 1, split='test')
    writer.add_text('Info', f'{ datetime.datetime.now() } Finish experiment')

    
    writer.add_hparams(exp_config, save_config['metric'], name=save_config['hparam_session_name'])
    writer.close()
