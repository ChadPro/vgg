from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import imagenet_224

datasets_map = {
    'imagenet_224' : imagenet_224
}

def get_dataset(name):
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return datasets_map[name]