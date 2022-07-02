import torch
import random
import numpy as np
import torch.nn as nn
from torch.nn import init
_MODELS = {}


def setup_seed(seed):
    if seed == -1:
        seed = random.randint(1,4096)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False
    else:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    return seed

def is_debugging():
  import sys
  gettrace = getattr(sys, 'gettrace', None)

  if gettrace is None:
    assert 0, ('No sys.gettrace')
  elif gettrace():
    return True
  else:
    return False

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(1.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def init_weights(modules, initialize):
    for module in modules():
        if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.ConvTranspose2d)
                or isinstance(module, nn.Linear)):
            if initialize == 'ortho':
                init.orthogonal_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            elif initialize == 'N02':
                init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            elif initialize in ['glorot', 'xavier']:
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            else:
                print('Init style not recognized...')
        elif isinstance(module, nn.Embedding):
            if initialize == 'ortho':
                init.orthogonal_(module.weight)
            elif initialize == 'N02':
                init.normal_(module.weight, 0, 0.02)
            elif initialize in ['glorot', 'xavier']:
                init.xavier_uniform_(module.weight)
            else:
                print('Init style not recognized...')
        else:
            pass

def register_model(cls=None, *, name=None):
  """A decorator for registering network classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered network with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)

def get_model(name):
  return _MODELS[name]
