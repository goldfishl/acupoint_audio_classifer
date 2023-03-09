import os
import random

# balance split dataset
def split_dataset(config):
    txt_files = [f for f in os.listdir(config['data_path']) if f.endswith('.txt')]

    labels = []
    for txt_file in txt_files:
        with open(os.path.join(config['data_path'], txt_file), 'r') as f:
            labels.append(f.read())
    
    data = {}
    for i, label in enumerate(labels):
        if label not in data:
            data[label] = []
        data[label].append(i)
    
    train_indices = []
    val_indices = []
    test_indices = []
    for label in data:
        random.shuffle(data[label])
        train_len = round(len(data[label]) * config['split_rate'][0])
        val_len = round(len(data[label]) * config['split_rate'][1])
        test_len = len(data[label]) - train_len - val_len
        train_indices += data[label][:train_len]
        val_indices += data[label][train_len:train_len+val_len]
        test_indices += data[label][train_len+val_len:]


    # Write the file lists to disk
    split_files = config['split_files'].values()
    split_indices = [train_indices, val_indices, test_indices]
    for split_file, split_indices in zip(split_files, split_indices):
        split_path = os.path.join(config['root_path'], split_file)
        split_indices = sorted(split_indices)
        with open(split_path, 'w') as f:
            for i in split_indices:
                f.write(f'{i}.wav\n')


def load_split(data_path, split_file):
    data = {}
    with open(split_file, 'r') as f:
        wav_files = []
        labels = []
        for line in f:
            wav_files.append(line.strip())
            label_file, _ = os.path.splitext(line)
            with open(os.path.join(data_path, label_file + '.txt'), 'r') as f:
                labels.append(f.read().strip())

    return (wav_files, labels)


def load_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
        labels = [line.strip() for line in lines]
    return labels

