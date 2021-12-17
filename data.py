import sklearn.datasets
import numpy as np
import torch

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

