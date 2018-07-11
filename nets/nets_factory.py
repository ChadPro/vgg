from nets import vgg11_nets
from nets import vgg11_LRN_nets


nets_map = {
    'vgg11_net_224' : vgg11_nets,
    'vgg11_LRN_net_224' : vgg11_LRN_nets
}

def get_network(name):
    if name not in nets_map:
        raise ValueError('Name of net unkonw %s' % name)
    return nets_map[name]