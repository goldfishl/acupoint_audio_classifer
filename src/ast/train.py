import torch
from torch import nn
import numpy as np
import os
import time
import datetime
from .ast_models import ASTModel
from .config import model_config, train_config, val_config, test_config, device
from .config import exp_config 
from src.utils import AverageMeter, d_prime, calculate_stats
from .dataset import AudioDataset
from tensorboardX import SummaryWriter


def train(audio_model, train_loader, val_loader, writer):
    torch.set_grad_enabled(True)

    global_step, epoch = 1, 1
    best_acc = 0
    exp_dir = exp_config['result_path']

    optimizer, warmup_scheduler = set_optimizer_scheduler(audio_model, writer)
    loss_fn = nn.BCEWithLogitsLoss()

    audio_model = audio_model.to(device)
    audio_model.train()
 
    while epoch < exp_config['n_epochs'] + 1:
        audio_model.train()
        for i, (audio_input, labels) in enumerate(train_loader):
                
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            audio_output = audio_model(audio_input, 'ft_cls')
            loss = loss_fn(audio_output, labels)

            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Learning Rate/base', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Learning Rate/head_mlp', optimizer.param_groups[1]['lr'], global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # warmup
            if global_step < exp_config['warmup_steps']:
                warmup_scheduler.step()

            global_step += 1

        # validate
        acc = validate(audio_model, val_loader, epoch, writer)

        if acc > best_acc:
            best_acc = acc
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        epoch += 1


def validate(audio_model, val_loader, epoch, writer):
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)

    # switch to evaluate mode
    audio_model.eval()


    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)
            labels = labels.to(device)

            # compute output
            audio_output = audio_model(audio_input, 'ft_cls')

            # compute the loss
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(audio_output, labels)

            predictions = audio_output.to('cpu').detach().sigmoid()
            A_predictions.append(predictions)
            A_targets.append(labels.to('cpu'))
            A_loss.append(loss.to('cpu').detach())

        predictions = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(predictions, target)


        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']

        middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)

        writer.add_scalar('Loss/val', loss, epoch)
        writer.add_scalar('Metrics/Accuracy', acc, epoch)
        writer.add_scalar('Metrics/mAUC', mAUC, epoch)
        writer.add_scalar('Metrics/mAP', mAP, epoch)
        writer.add_scalar('Metrics/Avg Precision', average_precision, epoch)
        writer.add_scalar('Metrics/Avg Recall', average_recall, epoch)
        writer.add_scalar('Metrics/d_prime', d_prime(mAUC), epoch)
        writer.add_pr_curve('Metrics/PR Curve', target, predictions, epoch)

    return acc


def set_optimizer_scheduler(audio_model, writer):

    # calculate statistics for the model
    writer.add_text('Text/model', 'Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    writer.add_text('Text/model', 'Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))


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
    
    writer.add_text('Text/hyperparameters', 'The mlp header uses {:d} x larger lr'.format(exp_config['head_lr']))
    writer.add_text('Text/model', 'Total mlp parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    writer.add_text('Text/model', 'Total base parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda step: (step / exp_config['warmup_steps']) * exp_config['lr'], 
                                                                               lambda step: (step / exp_config['warmup_steps']) * exp_config['lr'] * exp_config['head_lr']])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(exp_config['lrscheduler_start'], exp_config['lrscheduler_end'], exp_config['lrscheduler_step'])),gamma=exp_config['lrscheduler_gamma'])
    # writer.add_text('Text/hyperparameters', 'The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(exp_config['lrscheduler_start'], exp_config['lrscheduler_gamma'], exp_config['lrscheduler_step']))
    return optimizer, warmup_scheduler



if __name__ == '__main__':
    now = datetime.datetime.now()
    log_path = os.path.join(exp_config['result_path'], 'logs', now.strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    

    train_loader = torch.utils.data.DataLoader(
            AudioDataset(model_config, train_config),
            batch_size=train_config['batch_size'], shuffle=True, 
            num_workers=train_config['num_workers'], pin_memory=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
            AudioDataset(model_config, val_config),
            batch_size=val_config['batch_size'], shuffle=True, 
            num_workers=val_config['num_workers'], pin_memory=False)


    audio_model = ASTModel(label_dim=model_config['num_classes'], fshape=model_config['fshape'],
                        tshape=model_config['tshape'], fstride=model_config['fstride'], tstride=model_config['tstride'],
                        input_fdim=model_config['num_mel_bins'], input_tdim=model_config['target_length'],
                        model_size=model_config['model_size'], pretrain_stage=False,
                        load_pretrained_mdl_path=model_config['pretrained_mdl_path'])

    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)

    train(audio_model, train_loader, val_loader, writer)