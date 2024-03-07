import numpy as np
import argparse
import torch
import os
import glob

from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision import datasets, transforms

def load_raw_data(dir_data, rat_name): 
    trlPath = os.path.join(dir_data, rat_name, rat_name.lower() + '_trial_info.npy') 
    spkPath = os.path.join(dir_data, rat_name, rat_name.lower() + '_spike_data_binned.npy') 
    lfpPath = os.path.join(dir_data, rat_name, rat_name.lower() + '_lfp_data_sampled.npy')
    
    trial_info = np.load(trlPath) 
    spike_data = np.load(spkPath)
    lfp_data = np.load(lfpPath)
    lfp_data = np.swapaxes(lfp_data, 1, 2) 
    return trial_info, spike_data, lfp_data

def load_data(dir_data, num_modalities, start_times, window_len): 
    rat_names = ['barat', 'buchanan', 'mitt', 'stella', 'superchris'][:num_modalities]
    # rat_names = ['buchanan', 'stella', 'superchris'][:num_modalities]
    # rat_names = ['stella', 'superchris'][:num_modalities]
    trials_d, spikes_d, lfps_d, odorTarget_d = {},{},{},{}
    for view, rat_name in enumerate(rat_names):
        # different rat has different start_time and end_time
        start_time, end_time = start_times[view], start_times[view]+window_len
        trial_info, spike_data, lfp_data = load_raw_data(dir_data, rat_name) 
        trials_d[view] = trial_info
        odorTarget_d[view] = trials_d[view][:,3] - 1
        spikes_d[view] = spike_data[:,:,start_time:end_time]
        lfps_d[view] = lfp_data[:,:,start_time:end_time]
    return lfps_d, spikes_d, odorTarget_d, trials_d

class LFP(Dataset):
    """LFP Dataset."""

    def __init__(self, dir_data, train, num_views, start_times=[225,225,225,225,225], window_len=25, transform=None, target_transform=None):
        super().__init__()
        self.num_modalities = num_views
        self.dir_data = dir_data
        self.transform = transform
        self.target_transform = target_transform
        # load all the rats data in the time window (e.g. 225-250)
        self.start_times, self.window_len = start_times, window_len
        self.end_times = [(start_time + window_len) for start_time in start_times]
        self.lfps, self.spikes, self.odorTargets, self.trials = load_data(
            self.dir_data, self.num_modalities, self.start_times, self.window_len)
        self.num_trials_each_odor = [
            min([np.sum(labels==target_label) for labels in self.odorTargets.values()]) 
         for target_label in range(5)
        ]
        # self.num_data_total = sum(self.num_trials_each_odor) * (self.end_time - self.start_time)
        # only consider odor 0, 1, 2, 3
        # every 10 time points as a data point, so only 16 data points in every 25 time window
        self.sub_window_len = 10
        self.num_data_total = sum(self.num_trials_each_odor[:-1]) * (self.window_len - self.sub_window_len + 1)

    def __getitem__(self, index):
        # convert index to [trial, timepoint]
        trial = index // (self.window_len - self.sub_window_len + 1)
        for label in range(5):
            trial -= self.num_trials_each_odor[label]
            if trial < 0:
                trial += self.num_trials_each_odor[label]
                break
        time_point = index % (self.window_len - self.sub_window_len + 1)
        # get the item from lfp data
        images_dict = {
            "m%d" % m: torch.Tensor(
                self.lfps[m][self.odorTargets[m]==label][trial,:,time_point:(time_point+self.sub_window_len)]
            ).flatten() for m in range(self.num_modalities)
        }
        return images_dict, label

    def __len__(self):
        return self.num_data_total

class SPIKE(Dataset):
    """SPIKE Dataset."""

    def __init__(self, dir_data, train, num_views, start_times=[225,225,250,250,225], window_len=25, transform=None, target_transform=None):
        super().__init__()
        self.num_modalities = num_views
        self.dir_data = dir_data
        self.transform = transform
        self.target_transform = target_transform
        # load all the rats data in the time window (e.g. 225-250)
        self.start_times, self.window_len = start_times, window_len
        self.end_times = [(start_time + window_len) for start_time in start_times]
        self.lfps, self.spikes, self.odorTargets, self.trials = load_data(
            self.dir_data, self.num_modalities, self.start_times, self.window_len)
        self.num_trials_each_odor = [
            min([np.sum(labels==target_label) for labels in self.odorTargets.values()]) 
         for target_label in range(5)
        ]
        # self.num_data_total = sum(self.num_trials_each_odor) * (self.end_time - self.start_time)
        # only consider odor 0, 1, 2, 3
        # every 10 time points as a data point, so only 16 data points in every 25 time window
        self.sub_window_len = 10
        self.num_data_total = sum(self.num_trials_each_odor[:-1]) * (self.window_len - self.sub_window_len + 1)       

    def __getitem__(self, index):
        # convert index to [trial, timepoint]
        trial = index // (self.window_len - self.sub_window_len + 1)
        for label in range(5):
            trial -= self.num_trials_each_odor[label]
            if trial < 0:
                trial += self.num_trials_each_odor[label]
                break
        time_point = index % (self.window_len - self.sub_window_len + 1)
        # get the item from spike data
        images_dict = {
            "m%d" % m: torch.Tensor(
                self.spikes[m][self.odorTargets[m]==label][trial,:,time_point:(time_point+self.sub_window_len)]
            ).flatten() for m in range(self.num_modalities)
        }
        return images_dict, label

    def __len__(self):
        return self.num_data_total
