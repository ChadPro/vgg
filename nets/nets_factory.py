from nets import vgg_cifar_net
from nets import vgg11_nets
from nets import vgg11_LRN_nets

nets_map = {
    'vgg_cifar_net' : vgg_cifar_net,
    'vgg11_net_224' : vgg11_nets,
    'vgg11_LRN_net_224' : vgg11_LRN_nets
}

def get_network(name):
    if name not in nets_map:
        raise ValueError('Name of net unkonw %s' % name)
    return nets_map[name]