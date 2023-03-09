import os

# acupoint dataset config
acup_config = {
   'root_path' : os.path.join('data', 'acupoint'),
    'data_path' : os.path.join('data', 'acupoint', 'data'),
    'split_rate' : [0.8, 0.1, 0.1],
    'split_files' : {
        'train' : 'train.txt',
        'valid' : 'val.txt',
        'test' : 'test.txt'
    }
}

acup_config['data_path'] = os.path.join(acup_config['root_path'], 'data')
acup_config['split_files']['train'] = os.path.join(acup_config['root_path'], 'train.txt')
acup_config['split_files']['valid'] = os.path.join(acup_config['root_path'], 'val.txt')
acup_config['split_files']['test'] = os.path.join(acup_config['root_path'], 'test.txt')
acup_config['label_file'] = os.path.join(acup_config['root_path'], 'label.txt')




