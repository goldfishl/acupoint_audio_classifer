import os
from src.utils import wav2fbank, acup_config
import torch
from torch import nn, optim
import datetime


device = 'cuda' if torch.cuda.is_available() else 'cpu'
exp_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
num_mel_bins = 128

# ast audio classifier config
## model config
model_config = {
    'num_mel_bins' : num_mel_bins,
    'target_length' : 512,
    'num_classes' : 418,
}

# experiment config that will be saved in tensorboard
exp_config = {
    'model_name' : 'AudioTDNN',
    'batch_size' : 32,
    'lr' : 9e-4,
    'head_lr' : 1,  # head learning rate multiplier
    'weight_decay' : 5e-7,
    'lrscheduler_start' : 30, 
    'lrscheduler_step' : 1,
    'lrscheduler_end' : 1000,
    'lrscheduler_gamma' : 0.9,  # normal scheduler every epoch
    'n_epochs' : 50,
    'warmup_step' : 10,
    'warmup_end' : -1,  # set -1 to disable warmup
    'freq_mask' : 12,  # set 0 to disable freq_mask
    'time_mask' : 12,  # set 0 to disable time_mask
    'mixup' : 0.6,  # set 0 to disable mixup
    'noise' : True,
    'norm_mean' : -6.845978,
    'norm_std' : 5.5654526,
    'skip_norm' : True,
    'vad' : True,
    "vad_energy_threshold" : -5,  # log mel energy threshold
    "vad_bin_threshold" : 10,  # number of bins above energy threshold
    "vad_mask_threshold" : 5, # filter out silence frames shorter than this
    'loss_fn' : 'CrossEntropyLoss',
    'optimizer' : 'Adam',
    'clip_grad_norm' : 0.5,
}


# save config
save_config = {
    'log_dir' : os.path.join('logs', 'tdnn', exp_name),
    'hparam_log_dir' : os.path.join('logs', 'hyperparam', exp_name), # comment for tensorboard
    'hparam_session_name' : exp_name,  # comment for tensorboard
    'best_model_path' : os.path.join('models', f'{ exp_name }ssast_best.pth'),
    'worse_k' : 50,  # save the worse k recall class PR curve for analysis
    'metric' : {},  # record the result metrics for whole experiment in tensorboard
}



## train dataloader config
train_config = {
    'batch_size' : exp_config['batch_size'],
    'num_workers' : 8,
    'feature' : wav2fbank(model_config['num_mel_bins']),
    'data_path' : acup_config['data_path'],
    'split_file' : acup_config['split_files']['train'],
    'label_file' : acup_config['label_file'],
    'freq_mask' : exp_config['freq_mask'],  
    'time_mask' : exp_config['time_mask'],
    'mixup' : exp_config['mixup'],
    'norm_mean' : exp_config['norm_mean'],
    'norm_std' : exp_config['norm_std'],
    'skip_norm' : exp_config['skip_norm'],
    'noise' : exp_config['noise'],
    'vad' : exp_config['vad'],
    "vad_energy_threshold" : exp_config["vad_energy_threshold"],
    "vad_bin_threshold" : exp_config["vad_bin_threshold"],
    "vad_mask_threshold" : exp_config["vad_mask_threshold"],
}

## val dataloader config
val_config = train_config.copy()
val_config['batch_size'] = 64
val_config['split_file'] = acup_config['split_files']['valid']
val_config['freq_mask'] = 0
val_config['time_mask'] = 0
val_config['mixup'] = 0
val_config['noise'] = False

## test dataloader config
test_config = val_config.copy()
test_config['split_file'] = acup_config['split_files']['test']


def setup_training_params(model, writer):
    """
    Set up the optimizer, loss function, and learning rate scheduler.

    Args:
        model: PyTorch neural network model.
        writer: SummaryWriter for logging training information.

    Returns:
        tuple: A tuple of (optimizer, loss_fn, scheduler) for training the model.
    """
    # Calculate statistics for the model
    save_config['metric']['Hparam/model_params'] = sum(p.numel() for p in model.parameters()) / 1e6
    trainables = [p for p in model.parameters() if p.requires_grad]
    writer.add_text('Model', 'Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    # Set optimizer
    if exp_config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=exp_config['lr'], weight_decay=exp_config['weight_decay'], betas=(0.95, 0.999))

    # Set learning rate scheduler
    if exp_config['lrscheduler_start'] != -1:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, list(range(exp_config['lrscheduler_start'], exp_config['lrscheduler_end'], exp_config['lrscheduler_step'])), gamma=exp_config['lrscheduler_gamma'])

    # Set loss function
    if exp_config['loss_fn'] == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()

    return optimizer, scheduler, loss_fn