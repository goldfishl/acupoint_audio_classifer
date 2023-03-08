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





