import sklearn.datasets
import numpy as np
import os
from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

def data_process(dataset,batch_size):
    if dataset == 'two_moon':
        data, y = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)
    elif dataset == 'swiss_roll':
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
    elif dataset == '25gaussians':
        data = []
        for j in range(1000 // 25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    data.append(point)
        data = np.array(data[:batch_size], dtype='float32')
        np.random.shuffle(data)
        data /= 2.828
    data = torch.Tensor(data)
    return data

class InfiniteDataLoader(object):
    """docstring for InfiniteDataLoader"""

    def __init__(self, dataloader):
        super(InfiniteDataLoader, self).__init__()
        self.dataloader = dataloader
        self.data_iter = None

    def next(self):
        try:
            data = self.data_iter.next()
        except Exception:
            # Reached end of the dataset
            self.data_iter = iter(self.dataloader)
            data = self.data_iter.next()

        return data

    def __len__(self):
        return len(self.dataloader)

class RandomCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0]
            else np.random.randint(low=0,high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1]
            else np.random.randint(low=0,high=img.size[1] - size[1]))
        return transforms.functional.crop(img, j, i, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__
class LoadDataset(Dataset):
    def __init__(self, args,train=True):
        super(LoadDataset, self).__init__()
        self.train=train
        self.dataset=args.dataset
        self.norm_mean = [0.5,0.5,0.5]
        self.norm_std = [0.5,0.5,0.5]

        if args.dataset=='cifar10':
            self.transforms = []
        elif args.dataset== 'animeface':
            if self.train==True:
                self.transforms = [RandomCropLongEdge(), transforms.Resize(args.input_size)]
            else:
                self.transforms = [CenterCropLongEdge(), transforms.Resize(args.input_size)]
        if self.train==True:
            self.transforms+=[transforms.RandomHorizontalFlip()]

        self.transforms += [transforms.ToTensor(), transforms.Normalize(self.norm_mean, self.norm_std)]
        self.transforms = transforms.Compose(self.transforms)

        self.load_dataset()


    def load_dataset(self):

        if self.dataset == 'cifar10':
            self.data = datasets.CIFAR10(root='./data/cifar10',
                                train=self.train,
                                download=True)

        elif self.dataset =='animeface':
            mode = 'train' if self.train == True else 'valid'
            root = os.path.join("/dtu-compute/congen/animeface", mode)
            self.data = ImageFolder(root=root)

        else:
            raise NotImplementedError


    def __len__(self):

        num_dataset = len(self.data)
        return num_dataset


    def __getitem__(self, index):

        img, label = self.data[index]
        img, label = self.transforms(img), int(label)
        return img, label
