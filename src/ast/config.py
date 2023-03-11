import os
from src.utils import wav2fbank, acup_config
import torch
import datetime


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ast audio classifier config
## model config
model_config = {
    'num_mel_bins' : 128,
    'target_length' : 512,
    'num_classes' : 418,
    'fshape' : 128,
    'tshape' : 2,
    'fstride' : 128,
    'tstride' : 1,
    'model_size' : 'base',
    'pretrained_mdl_path' : os.path.join('models', 'SSAST-Base-Frame-400.pth'),
}

# experiment config that will be saved in tensorboard
exp_config = {
    'model_name' : 'SSAST-Base-Frame-400',
    'batch_size' : 32,
    'lr' : 4e-4,
    'head_lr' : 1.5,  # head learning rate multiplier
    'weight_decay' : 5e-7,
    'lrscheduler_start' : 5, 
    'lrscheduler_step' : 1,
    'lrscheduler_end' : 1000,
    'lrscheduler_gamma' : 0.8,  # normal scheduler every epoch
    'n_epochs' : 30,
    'warmup_step' : 1,
    'warmup_end' : 1000,  # set -1 to disable warmup
    'freq_mask' : 48,  # set 0 to disable freq_mask
    'time_mask' : 48,  # set 0 to disable time_mask
    'mixup' : 0.6,  # set 0 to disable mixup
    'noise' : True,
    'norm_mean' : -6.845978,
    'norm_std' : 5.5654526,
    'skip_norm' : False,
}

run_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# save config
save_config = {
    'log_dir' : os.path.join('logs', 'ssast', run_name),
    'hparam_session_name' : 'hyperparam'+run_name, # comment for tensorboard
    'best_model_path' : os.path.join('models', f'{ run_name }ssast_best.pth'),
    'worse_k' : 50,  # save the worse k recall class PR curve for analysis
    'metric' : {},  # record the result metrics for whole experiment in tensorboard
}
test_confusion_matrix_path = os.path.join(save_config['log_dir'], 'test_confusion_matrix.csv')


## train dataloader config
train_config = {
    'batch_size' : exp_config['batch_size'],
    'num_workers' : 8,
    'feature' : wav2fbank(model_config['num_mel_bins'],
                          compliance='kaldi'),
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