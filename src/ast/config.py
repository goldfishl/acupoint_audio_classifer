import os
from src.utils import wav2fbank, acup_config
import torch


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

# experiment config
exp_config = {
    'head_lr' : 1,  # head learning rate multiplier
    'lr' : 2.5e-4,
    'weight_decay' : 5e-7,
    # 'lrscheduler_start' : 5,
    # 'lrscheduler_step' : 1,
    # 'lrscheduler_end' : 1000,
    # 'lrscheduler_gamma' : 0.85,
    'n_epochs' : 30,
    'warmup' : True,
    'warmup_steps' : 1000,
    'result_path' : os.path.join('results', 'ssast'),
    'n_print_steps' : 100,
}

## train data config
train_config = {
    'batch_size' : 32,
    'num_workers' : 8,
    'feature' : wav2fbank(model_config['num_mel_bins'],
                          compliance='kaldi'),
    'data_path' : acup_config['data_path'],
    'split_file' : os.path.join(acup_config['root_path'], 'train.txt'),
    'label_file' : os.path.join(acup_config['root_path'], 'label.txt'),
    'freq_mask' : 48,  # set 0 to disable freq_mask
    'time_mask' : 48,  # set 0 to disable time_mask
    'mixup' : 0.6,  # set 0 to disable mixup
    'norm_mean' : -6.845978,
    'norm_std' : 5.5654526,
    'skip_norm' : False,
    'noise' : True,
}

## val data config
val_config = train_config.copy()
val_config['batch_size'] = 64
val_config['split_file'] = os.path.join(acup_config['root_path'], 'val.txt')
val_config['freq_mask'] = 0
val_config['time_mask'] = 0
val_config['mixup'] = 0
val_config['noise'] = False

## test data config
test_config = train_config.copy()
test_config['split_file'] = os.path.join(acup_config['root_path'], 'test.txt')