from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def print_tensor_shape(tensor, string):

# input: tensor and string to describe it

    if __debug__:
        print('DEBUG ' + string, tensor.get_shape())

def inference(images, size, keep_prob=1, batch_size=1, regularization_constant=0.0):

#   input: tensor of images
#   output: tensor of computed logits

# resize the image tensors to add the number of channels, 1 in this case
# required to pass the images to various layers upcoming in the graph
    #print("Image size:", size)
    #num_channels = size[0], depth = size[0], height = size[1], width = size[2], num_channels = size[3]
    print_tensor_shape(images, "images")
# Convolution layer
    with tf.name_scope('Matmul1'):

        W_matmul1 = tf.Variable(tf.truncated_normal([size[0]*size[1]*size[2]*size[3], 4096],stddev=0.1,
                     dtype=tf.float32),name='W_matmul1')
        print_tensor_shape( W_matmul1, 'W_matmul1 shape')


        matmul1_op = tf.matmul(images, W_matmul1)

        print_tensor_shape( matmul1_op, 'matmul1_op shape')

        W_bias1 = tf.Variable( tf.zeros([4096], dtype=tf.float32), 
                          name='W_bias1')
        print_tensor_shape( W_bias1, 'W_bias1 shape')

        bias1_op = matmul1_op + W_bias1
        print_tensor_shape( bias1_op, 'bias1_op shape')

        relu1_op = tf.nn.relu( bias1_op, name='relu1_op' )
        print_tensor_shape( relu1_op, 'relu1_op shape')

    with tf.name_scope('Matmul2'):

        W_matmul2 = tf.Variable(tf.truncated_normal([4096, 2048],stddev=0.1,
                     dtype=tf.float32),name='W_matmul2')
        print_tensor_shape( W_matmul2, 'W_matmul2 shape')


        matmul2_op = tf.matmul(relu1_op, W_matmul2)

        print_tensor_shape( matmul2_op, 'matmul2_op shape')

        W_bias2 = tf.Variable( tf.zeros([2048], dtype=tf.float32), 
                          name='W_bias2')
        print_tensor_shape( W_bias2, 'W_bias2 shape')

        bias2_op = matmul2_op + W_bias2
        print_tensor_shape( bias2_op, 'bias2_op shape')

        relu2_op = tf.nn.relu( bias2_op, name='relu1_op' )
        print_tensor_shape( relu2_op, 'relu2_op shape')

    with tf.name_scope('Matmul3'):

        W_matmul3 = tf.Variable(tf.truncated_normal([2048, 1024],stddev=0.1,
                     dtype=tf.float32),name='W_matmul3')
        print_tensor_shape( W_matmul3, 'W_matmul3 shape')


        matmul3_op = tf.matmul(relu2_op, W_matmul3)

        print_tensor_shape( matmul3_op, 'matmul3_op shape')

        W_bias3 = tf.Variable( tf.zeros([1024], dtype=tf.float32), 
                          name='W_bias3')
        print_tensor_shape( W_bias3, 'W_bias3 shape')

        bias3_op = matmul3_op + W_bias3
        print_tensor_shape( bias3_op, 'bias3_op shape')

        relu3_op = tf.nn.relu( bias3_op, name='relu1_op' )
        print_tensor_shape( relu3_op, 'relu3_op shape')

    with tf.name_scope('Matmul4'):

        W_matmul4 = tf.Variable(tf.truncated_normal([1024, 512],stddev=0.1,
                     dtype=tf.float32),name='W_matmul4')
        print_tensor_shape( W_matmul4, 'W_matmul3 shape')


        matmul4_op = tf.matmul(relu3_op, W_matmul4)

        print_tensor_shape( matmul4_op, 'matmul4_op shape')

        W_bias4 = tf.Variable( tf.zeros([512], dtype=tf.float32), 
                          name='W_bias4')
        print_tensor_shape( W_bias4, 'W_bias4 shape')

        bias4_op = matmul4_op + W_bias4
        print_tensor_shape( bias4_op, 'bias4_op shape')

        relu4_op = tf.nn.relu( bias4_op, name='relu1_op' )
        print_tensor_shape( relu4_op, 'relu4_op shape')

        drop_op = tf.nn.dropout( relu4_op, keep_prob )
        print_tensor_shape( drop_op, 'drop_op shape' )

    with tf.name_scope('Matmul5'):

        W_matmul5 = tf.Variable(tf.truncated_normal([512, 2],stddev=0.1,
                     dtype=tf.float32),name='W_matmul5')
        print_tensor_shape( W_matmul5, 'W_matmul5 shape')

        matmul5_op = tf.matmul(drop_op, W_matmul5)
        print_tensor_shape( matmul5_op, 'matmul5_op shape')

        W_bias5 = tf.Variable( tf.zeros([2], dtype=tf.float32), 
                          name='W_bias5')
        print_tensor_shape( W_bias5, 'W_bias5 shape')

        bias5_op = matmul5_op + W_bias5
        print_tensor_shape( bias5_op, 'bias5_op shape')


    #Regularization of all the weights in the network for the loss function
    with tf.name_scope('Regularization'):
      Reg_constant = tf.constant(regularization_constant)
      reg_op = tf.nn.l2_loss(W_matmul1) + tf.nn.l2_loss(W_matmul2)  + tf.nn.l2_loss(W_matmul3) + tf.nn.l2_loss(W_matmul4) + tf.nn.l2_loss(W_matmul5)
      reg_op = reg_op*Reg_constant
      tf.summary.scalar('reg_op', reg_op)

    return bias5_op + reg_op

def evaluation(logits, labels):
    #return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1), tf.argmax(labels,1)), tf.float32))
    values, indices = tf.nn.top_k(labels, 1);
    correct = tf.reshape(tf.nn.in_top_k(logits, tf.cast(tf.reshape( indices, [-1 ] ), tf.int32), 1), [-1] )
    print_tensor_shape( correct, 'correct shape')
    return tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

def training(loss, learning_rate, decay_steps, decay_rate):
    # input: loss: loss tensor from loss()
    # input: learning_rate: scalar for gradient descent
    # output: train_op the operation for training

#    Creates a summarizer to track the loss over time in TensorBoard.

#    Creates an optimizer and applies the gradients to all trainable variables.

#    The Op returned by this function is what must be passed to the
#    `sess.run()` call to cause the model to train.

  # Add a scalar summary for the snapshot loss.

  # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

  # create learning_decay
    lr = tf.train.exponential_decay( learning_rate,
                                     global_step,
                                     decay_steps,
                                     decay_rate, staircase=True )

    tf.summary.scalar('2learning_rate', lr )

  # Create the gradient descent optimizer with the given learning rate.
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(lr)

  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def loss(logits, labels):
    
    print_tensor_shape( logits, 'logits shape')
    print_tensor_shape( labels, 'labels shape')

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy')

    loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

    return loss
