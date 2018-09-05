# Copyright 2018 The LongYan. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division

import numpy as np
import heapq

def acc_top5(net_output, label_input):
    
    out_batch_size, out_classes_num = net_output.shape
    input_batch_size, input_classes_num = label_input.shape

    acc_mark = np.zeros(out_batch_size)
    batch_label = np.argmax(label_input, 1)
    for i in range(out_batch_size):
        net_label = batch_label[i]

        top5_list = heapq.nlargest(5, net_output[i])
        net_out = []
        for top_value in top5_list:
            for j,out_value in enumerate(net_output[i]):
                if top_value == out_value:
                    net_out.append(j) 

        if net_label in net_out:
            acc_mark[i] = 1


    batch_acc_mean = np.mean(acc_mark)
    return str(batch_acc_mean * 100.) + "%"

def acc_top1(net_output, label_input):
    
    batch_label = np.argmax(label_input, 1)
    batch_output = np.argmax(net_output, 1)

    result_batch = np.equal(batch_label, batch_output)
    result_mean = np.mean(result_batch)

    return str(result_mean * 100.) + "%"