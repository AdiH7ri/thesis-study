import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import re 
from pathlib import Path

DIR_PATH = Path(__file__).parent

VID_SOURCE = DIR_PATH.parent / 'left-cam'
SAVEPATH = DIR_PATH / 'extracted-frames'

class VidData3D(data.Dataset):
    '''
    
    Custom Dataset object for loading the video files
    
    '''

    def __init__(self, data_file, transform=None):

        self.data_file = data_file
        self.transform = transform
        self.index_map = {}
        index = 0

        for pid in data_file:  # First dimension
                    rounds = data_file[pid]
                    for round in rounds:  # Second dimension
                        for i,_ in enumerate(data_file[pid][round]['labels']):
                            self.index_map[index] = (pid, round, i)
                            index += 1

    def __len__(self):

        return len(self.index_map)

    def read_images(self, p_id, r, idx, use_transform):

        X = []

        for frame_image_name in self.data_file[p_id][r]['frames'][idx]:
            
            frame_path =  SAVEPATH / 'size-256x256' / p_id / r / frame_image_name
            image = Image.open(frame_path).convert('L')

            if use_transform is not None:
                image = use_transform(image)

            X.append(image.squeeze_(0))

        X = torch.stack(X, dim=0)

        return X
    
    def read_labels(self, p_id, r, idx):
        # labels = []

        # for label in self.data_file[p_id][r]['labels'][idx]:
        #     labels.extend(label)

        return(self.data_file[p_id][r]['labels'][idx])
    
    def __getitem__(self, index):

        # Select sample
        (pid, round, idx) = self.index_map[index]

        # Load data
        X = self.read_images(pid, round, idx, self.transform).unsqueeze_(0)  # (input) spatial images
        y = torch.FloatTensor([float(self.read_labels(pid, round, idx))])                             # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y

class VidData2D(data.Dataset):
    '''
    
    Custom Dataset object for loading the video files
    
    '''

    def __init__(self, data_file, transform=None):

        self.data_file = data_file
        self.transform = transform
        self.index_map = {}
        index = 0

        for pid in data_file:  # First dimension
                    rounds = data_file[pid]
                    for round in rounds:  # Second dimension
                        for i,_ in enumerate(data_file[pid][round]['labels']):
                            self.index_map[index] = (pid, round, i)
                            index += 1

    def __len__(self):

        return len(self.index_map)

    def read_images(self, p_id, r, idx, use_transform):

        X = []

        for frame_image_name in self.data_file[p_id][r]['frames'][idx]:
            
            frame_path =  SAVEPATH / 'size-256x256' / p_id / r / frame_image_name
            image = Image.open(frame_path)

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)

        X = torch.stack(X, dim=0)

        return X
    
    def read_labels(self, p_id, r, idx):
        # labels = []

        # for label in self.data_file[p_id][r]['labels'][idx]:
        #     labels.extend(label)

        return(self.data_file[p_id][r]['labels'][idx])
    
    def __getitem__(self, index):

        # Select sample
        (pid, round, idx) = self.index_map[index]

        # Load data
        X = self.read_images(pid, round, idx, self.transform) # (input) spatial images
        y = torch.FloatTensor([float(self.read_labels(pid, round, idx))])                             # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y
    
def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def generate_chunks(frames_list, window_length):

    for i in range(0, len(frames_list), window_length):  
        yield frames_list[i:i + window_length] 


def return_frame_chunks(frames_path, window_length):

    frames = sorted_nicely([f.name for f in frames_path.iterdir()])

    frames_sliced = frames[(len(frames) % window_length):]

    return [ i for i in generate_chunks(frames_sliced, window_length)]


def return_stat_dict(source_path, p_ids, window_length):
    
    stat_dict = {}

    for p in p_ids:

        rounds = [f.name for f in os.scandir(source_path / p) if f.is_dir()] 

        round_stat = {}

        for r in rounds:
            
            vid_stat = {}    

            frames_path = source_path / p / r

            vid_stat['frames'] = return_frame_chunks(frames_path, window_length)

            if p[0] == 'N':
                vid_stat['labels'] = np.zeros( len(return_frame_chunks(frames_path, window_length)) ).tolist()
            else:
                vid_stat['labels'] = np.ones( len(return_frame_chunks(frames_path, window_length)) ).tolist()
            
            round_stat[r] = vid_stat

        stat_dict[p] = round_stat
    
    return stat_dict

# def generate_stratified_train_test_pair_ids(Seed = s):
     
