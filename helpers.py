import numpy as np
import os
import cv2
from enum import Enum
from tensorflow.python.ops.init_ops import VarianceScaling
import collections

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def downsample(img, height, width):
    return cv2.resize(img, dsize = (width, height), interpolation=cv2.INTER_LINEAR)

def rgb2gray(image):
  return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def preprocess(img, height = 84, width = 84):
    return cv2.resize(rgb2gray(img)/255., (width, height))

def get_save_path():
    path = './model'
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except:
        print('Creare directorului a esuat')
        path = './'
    return path

class Optimizers(Enum):
    SGD = 1
    RMSprop = 2
    Adam = 3
    Adadelta = 4
    Adagrad = 5
    Adamax = 6
    Nadam = 7
    Ftrl = 8

class Models(Enum):
    Classic = 1
    New = 2

def lecun_normal(seed=None):
		return VarianceScaling(scale=1., mode='fan_in', distribution='truncated_normal', seed=seed)
def lecun_uniform(seed=None):
	return VarianceScaling(scale=1., mode='fan_in', distribution='uniform', seed=seed)

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)

