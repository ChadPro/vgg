# -- coding: utf-8 --

import tensorflow as tf
import numpy as np
import time
from nets import nets_factory
from datasets import datasets_factory

################
# Train param  #
################
# Default param
IMAGE_SIZE = 224
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
# Adjust params
tf.app.flags.DEFINE_string('net_chose', 'vgg11_net_224', 'Chose which net.')
tf.app.flags.DEFINE_float('learning_rate_base', 0.01, 'Initial learning base rate.')
tf.app.flags.DEFINE_float('learning_rate_decay', 0.99, 'Learning rate decay.' )
tf.app.flags.DEFINE_integer('learning_decay_step', 500, 'Learning rate decay step.')
tf.app.flags.DEFINE_integer('total_steps', 300000, 'Total train steps.')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Data batch size.')
tf.app.flags.DEFINE_float('gpu_fraction', 0.7, 'How to use gpu.')
tf.app.flags.DEFINE_integer('num_classes', 1000, 'Classes num.')
tf.app.flags.DEFINE_bool('fine_tune', False, 'Is fine_tune work.')

tf.app.flags.DEFINE_string('log_dir', './board_log', 'Log file saved.')
tf.app.flags.DEFINE_string('dataset', 'imagenet_224', 'Chose dataset in dataset_factory.')
tf.app.flags.DEFINE_string('train_data_path', '', 'Dataset path for train.')
tf.app.flags.DEFINE_string('val_data_path', '', 'Dataset path for val.')
tf.app.flags.DEFINE_string('train_model_dir', './model/model.ckpt', 'Directory where checkpoints are written to.')
tf.app.flags.DEFINE_string('restore_model_dir', '', 'Restore model.')
FLAGS = tf.app.flags.FLAGS

def train():
    #1. Get vgg net
    vgg_net = nets_factory.get_network(FLAGS.net_chose)
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3], name='net_x_input')
    y_ = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.num_classes], name='net_y_input')
    label_y_ = tf.argmax(y_, 1)
    isTrainNow = tf.placeholder(tf.bool, name='isTrainNow')

    #2. Forward propagation
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y, restore_var_list = vgg_net.vgg_net(x, num_classes=FLAGS.num_classes, is_training=isTrainNow, train='train', \
                        fine_tune=FLAGS.fine_tune, regularizer=regularizer)
    output_y = tf.argmax(y, 1)
    global_step = tf.Variable(0, trainable=False)
    
    #3. Calculate cross_entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    total_loss = cross_entropy_mean + tf.add_n(tf.get_collection('regular_losses'))

    #4. Back propagation
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate_base, global_step, FLAGS.learning_decay_step, FLAGS.learning_rate_decay)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

    #5. Calculate val accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #6. Tensorboard summary and Saver persistent
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('val_acc', accuracy)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.log_dir, tf.get_default_graph())
    model_saver = tf.train.Saver(restore_var_list)

    #7. Get Dataset
    vgg_dataset = datasets_factory.get_dataset(FLAGS.dataset)
    input_X, input_Y, testtest = vgg_dataset.inputs(FLAGS.train_data_path, \
                                                    FLAGS.val_data_path, \
                                                    'train', \
                                                    FLAGS.batch_size,
                                                    None)
    input_X_val, input_Y_val, testtest_val = vgg_dataset.inputs(FLAGS.train_data_path, \
                                                    FLAGS.val_data_path, \
                                                    'val', \
                                                    FLAGS.batch_size,
                                                    None)

    #8. Start Train Session
    init_variables = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
    with tf.Session(config=config) as sess:
        sess.run(init_variables)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        startTime = time.time()

        if len(FLAGS.restore_model_dir) > 0:
            print "#####=============> Restore Model : "+str(FLAGS.restore_model_dir)
            model_saver.restore(sess, FLAGS.restore_model_dir)

        for i in range(FLAGS.total_steps):
            x_input, y_input = sess.run([input_X, input_Y])
            _, loss_value, step = sess.run([train_step, total_loss, global_step], feed_dict={x:x_input, y_:y_input, isTrainNow:True})

            if i % 30 == 0:
                learning_rate_now = FLAGS.learning_rate_base * (FLAGS.learning_rate_decay**(step / FLAGS.learning_decay_step))
                x_input_val, y_input_val = sess.run([input_X_val, input_Y_val])
                summary_str, result, outy, outy_ = sess.run([merged, accuracy, output_y, label_y_], feed_dict={x:x_input_val, y_:y_input_val, isTrainNow:False})
                writer.add_summary(summary_str, i)

                acc = result * 100.0
                accStr = str(acc) + "%"

                run_time = time.time() - startTime
                run_time = run_time / 60

                work_type = 'train'
                if FLAGS.fine_tune:
                    work_type = 'fine tune'
                print("########### " + work_type +" ###############")
                print("############ step : %d ################"%step)
                print("   learning_rate = %g                    "%learning_rate_now)
                print("   lose(batch)   = %g                    "%loss_value)
                print("   accuracy      = " + accStr)
                print("   train run     = %d min"%run_time)
                print" output : ", outy[0:10]
                print" label : ", outy_[0:10]
                print(" ")
                print(" ")

            if i % 500 == 0:
                model_saver.save(sess, FLAGS.train_model_dir)

        writer.close()
        durationTime = time.time() - startTime
        minuteTime = durationTime / 60
        print "To train the model, we use %d minutes." %minuteTime

        coord.request_stop()
        coord.join(threads)

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()