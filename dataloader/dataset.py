import os
import json
import random
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from runner.utils import get_config

random.seed(50)

# CSI dataset loader for Meta-Learning
class CSIDataset(data.Dataset):
    def __init__(self, data_path, num_support, num_query, win_size=10, mode='train', amp=True):
        self.mode = mode
        self.win_size = win_size
        self.config = get_config('config.yaml')
        self.labels = self.config['activity_labels']
        self.amp = amp

        # Read CSI file and convert to dataframe
        self.data_df = dict()
        
        for atv in self.labels:
            f_path = os.path.join(data_path, atv + '.csv')
            self.data_df[atv] = pd.read_csv(f_path)

        # Generated window size CSI data
        self.min_num_wd = int(1e9)  # for balanced dataset class
        self.data = self.generate_windows()

        # Variables for episodic learning
        self.num_support = num_support
        self.num_query = num_query
        self.num_episode = self.min_num_wd // (self.num_support + self.num_query)

    def generate_windows(self):
        win_dict = dict()
        for atv in self.data_df.keys():
            windows = list()
            df = self.data_df[atv]
            
            num_win = len(df) // self.win_size

            self.min_num_wd = min(self.min_num_wd, num_win)

            for i in range(num_win):
                wd = df.iloc[i*self.win_size:i*self.win_size + self.win_size, 2:]
                wd = wd.astype(complex)
            
                if self.amp is True:
                    wd = wd.apply(lambda x: x.abs())
                
                wd = wd.to_numpy()
                windows.append(wd)
            
            win_dict[atv] = np.array(windows)
        return win_dict
        

    def __len__(self):
        return self.num_episode

    def __getitem__(self, idx):
        # idx means index of episode
        sample = dict()

        for label in self.labels:
            sample[label] = dict()

            # support set
            support_indices = random.sample(range(self.min_num_wd), self.num_support)
            support_set = [self.data[label][i] for i in support_indices]
            sample[label]['support'] = np.array(support_set)

            # query set
            query_indices = list(set(range(self.min_num_wd)) - set(support_indices))
            query_indices = random.sample(query_indices, self.num_query)
            query_set = [self.data[label][i] for i in query_indices]
            sample[label]['query'] = np.array(query_set)

        return sample
    

# Supervised learning dataset
class SVLDataset(data.Dataset):
    def __init__(self, data_path, win_size=10, mode='train', train_proportion=0.8, amp=True, gan='false'):
        self.mode = mode
        self.win_size = win_size
        self.config = get_config('config.yaml')
        self.labels = self.config['activity_labels']
        self.amp = amp
        self.train_proportion = train_proportion
        self.gan = gan

        # Read CSI file and convert to dataframe
        self.data_df = dict()
        self.min_data_len = int(1e9)

        for atv in self.labels:
            f_path = os.path.join(data_path, atv + '.csv')
            self.data_df[atv] = pd.read_csv(f_path)
            self.min_data_len = min(self.min_data_len, len(self.data_df[atv]))


        # Generated window size CSI data
        self.min_num_wd = self.min_data_len // self.win_size  # for balanced dataset class
        
        self.num_train = int(self.min_num_wd * self.train_proportion)
        self.num_test = self.min_num_wd - self.num_train
        
        self.data_x, self.data_y = self.generate_windows()

    def generate_windows(self):
        win_data_x = list()
        data_y = list()
        atvs = list()
        if self.gan == 'false':
            atvs = list(self.data_df.keys())
        elif self.gan == 'true':
            atvs = [self.config['label_A'], self.config['label_B']]

        for idx, atv in enumerate(atvs):
            windows = list()
            df = self.data_df[atv]

            # # One-hot-encoding
            # y_label = np.zeros(len(self.labels))
            # y_label[idx] = 1.0
            y_label = idx

            for i in range(self.min_num_wd):
                wd = df.iloc[i*self.win_size:i*self.win_size + self.win_size, 2:]
                wd = wd.astype(complex)
            
                if self.amp is True:
                    wd = wd.apply(lambda x: x.abs())
                
                wd = wd.to_numpy()
                windows.append(wd)
            
            if self.mode == 'train':
                windows = windows[:self.num_train]
                data_y.extend([y_label for _ in range(len(windows))])
            elif self.mode == 'test':
                windows = windows[self.num_train:]
                data_y.extend([y_label for _ in range(len(windows))])
            
            if self.gan == 'false':
                win_data_x.extend(windows)
            elif self.gan == 'true':
                win_data_x.append(windows)
        
        if self.gan == 'false':
            return np.array(win_data_x), np.array(data_y)
        elif self.gan == 'true':
            return np.array(win_data_x[0]), np.array(win_data_x[1])

    def __len__(self):
        if self.gan == 'false':
            return len(self.data_x)
        elif self.gan == 'true':
            return min(len(self.data_x), len(self.data_y))

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]
       
    
# Noise dataset
# Read npy files
# amplitude noise data
class NoiseDataset(data.Dataset):
    def __init__(self, win_size=10, mode='train'):
        self.mode = mode
        self.win_size = win_size
        self.config = get_config('config.yaml')
        self.data_path = self.config['train_noise_path']
        if mode == 'test':
            self.data_path = self.config['test_noise_path']
        self.label_A = self.config['label_A']
        self.label_B = self.config['label_B']

        # Read npy files
        self.data_A = np.load(os.path.join(self.data_path, self.label_A+'_noise.npy'))
        self.data_B = np.load(os.path.join(self.data_path, self.label_B+'_noise.npy'))

    def __len__(self):
        return min(len(self.data_B), len(self.data_A))

    def __getitem__(self, idx):
        return self.data_A[idx], self.data_B[idx]
    

class FakeDataset(data.Dataset):
    def __init__(self, data_path, win_size=10):
        self.win_size = win_size
        self.config = get_config('config.yaml')
        self.data_path = data_path
        self.labels = self.config['activity_labels']
        self.label = 'sit'

        # Read npy files
        self.data = np.load(os.path.join(self.data_path, f'fake_{self.label}.npy'))

        idx = self.labels.index(self.label)
        self.y_label = np.zeros(len(self.labels))
        self.y_label[idx] = 1.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.y_label
    

if __name__ == '__main__':
    data = CSIDataset('../csi_dataset/domain_B',5,10,win_size=10,mode='train')
    print(data.__len__())
    print(data.__getitem__(0)['empty']['support'].shape)