import torch
from torch import nn
import numpy as np
import pickle
import os
import time
import datetime
from .ast_models import ASTModel
from .config import model_config, train_config, val_config, test_config, device
from .config import exp_config 
from src.utils import AverageMeter, d_prime, calculate_stats
from .dataset import AudioDataset
from tensorboardX import SummaryWriter


def train(audio_model, train_loader, val_loader, exp_config, writer):
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = exp_config['result_path']

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP,
                time.time() - start_time])
        with open(os.path.join(exp_dir,'progress.pkl'), 'wb') as f:
            pickle.dump(progress, f)


    audio_model = audio_model.to(device)
    
    # print model info
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

    writer.add_text('Text/hyperparameters', 'The mlp header uses {:d} x larger lr'.format(exp_config['head_lr']))
    optimizer = torch.optim.Adam([{'params': base_params, 'lr': exp_config['lr']}, {'params': mlp_params, 'lr': exp_config['lr'] * exp_config['head_lr']}], weight_decay=5e-7, betas=(0.95, 0.999))
    mlp_lr = optimizer.param_groups[1]['lr']
    lr_list = [exp_config['lr'], mlp_lr]

    writer.add_text('Text/model', 'Total mlp parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    writer.add_text('Text/model', 'Total base parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(exp_config['lrscheduler_start'], exp_config['lrscheduler_end'], exp_config['lrscheduler_step'])),gamma=exp_config['lrscheduler_gamma'])

    main_metrics = 'acc'
    loss_fn = nn.BCEWithLogitsLoss()

    writer.add_text('Text/hyperparameters', 'The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(exp_config['lrscheduler_start'], exp_config['lrscheduler_gamma'], exp_config['lrscheduler_step']))

    epoch += 1

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([exp_config['n_epochs'], 8])
    audio_model.train()
    while epoch < exp_config['n_epochs'] + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (audio_input, labels) in enumerate(train_loader):
                
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 1000 and global_step % 50 == 0 and exp_config['warmup'] == True:
                for group_id, param_group in enumerate(optimizer.param_groups):
                    warm_lr = (global_step / 1000) * lr_list[group_id]
                    param_group['lr'] = warm_lr
                    print('warm-up learning rate is {:f}'.format(param_group['lr']))

            audio_output = audio_model(audio_input, 'ft_cls')
            loss = loss_fn(audio_output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/audio_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/audio_input.shape[0])

            print_step = global_step % exp_config['n_print_steps'] == 0
            early_print_step = epoch == 0 and global_step % (exp_config['n_print_steps']/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.avg:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("yuan training diverged...")
                    torch.save(audio_model.state_dict(), "%s/models/nan_audio_model.pth" % (exp_dir))
                    torch.save(optimizer.state_dict(), "%s/models/nan_optim_state.pth" % (exp_dir))
                    with open(exp_dir + '/audio_input.npy', 'wb') as f:
                        np.save(f, audio_input.cpu().detach().numpy())
                    np.savetxt(exp_dir + '/audio_output.csv', audio_output.cpu().detach().numpy(), delimiter=',')
                    np.savetxt(exp_dir + '/labels.csv', labels.cpu().detach().numpy(), delimiter=',')
                    print('audio output and label saved for debugging.')
                    #return

            end_time = time.time()
            global_step += 1

        print('start validation')
        stats, valid_loss = validate(audio_model, val_loader, epoch, exp_dir)



        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']

        middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)

        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("Avg Precision: {:.6f}".format(average_precision))
        print("Avg Recall: {:.6f}".format(average_recall))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        result[epoch-1, :] = [acc, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg, valid_loss, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        # save every models
        # torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        
        if len(train_loader.dataset) > 2e5:
            torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            print('adaptive learning rate scheduler step')
            scheduler.step(mAP)
        else:
            print('normal learning rate scheduler step')
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[1]['lr']))

        with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        # break if lr too small
        # if optimizer.param_groups[0]['lr'] < args.lr/64 and epoch > 10:
        #     break

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()

def validate(audio_model, val_loader, epoch, exp_dir):
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            # compute output
            audio_output = audio_model(audio_input, 'ft_cls')
            predictions = audio_output.to('cpu').detach()


            A_predictions.append(predictions)
            A_targets.append(labels)

            # compute the loss
            labels = labels.to(device)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(audio_output, labels)

            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()



        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)

        # save the prediction here
        if os.path.exists(exp_dir+'/predictions') == False:
            os.mkdir(exp_dir+'/predictions')
            np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
        np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')

    return stats, loss

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

    train(audio_model, train_loader, val_loader, exp_config, writer)