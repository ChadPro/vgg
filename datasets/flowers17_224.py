import numpy as np
import tensorflow as tf
import time
from os import walk
import cv2

# Image 
IMG_SIZE = 224
IMG_CHANNELS = 3

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,features={
        'label':tf.FixedLenFeature([],tf.int64),
        'image_raw':tf.FixedLenFeature([],tf.string)
    })
    image = tf.decode_raw(features['image_raw'],tf.uint8)
    label = tf.cast(features['label'],tf.int32)
    # image.set_shape([IMG_WIDTH*IMG_HEIGHT*IMG_CHANNELS])
    image = tf.reshape(image,[IMG_SIZE,IMG_SIZE,IMG_CHANNELS])
    return image, label

def inputs(train_path, val_path, data_set,batch_size,num_epochs):
    if not num_epochs:
        num_epochs = None
    if data_set == 'train':
        file = train_path
    else:
        file = val_path
    with tf.name_scope('tfrecord_input') as scope:
        filename_queue = tf.train.string_input_producer([file], num_epochs=num_epochs)
        image,label = read_and_decode(filename_queue)
        images,labels = tf.train.shuffle_batch([image,label], batch_size=batch_size, num_threads=64, capacity=5000, min_after_dequeue=3000)

        ll = tf.expand_dims(labels, 1)
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        concated = tf.concat([indices, ll], 1)
        onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 1000]), 1.0, 0.0)

    return images, onehot_labels, labels
