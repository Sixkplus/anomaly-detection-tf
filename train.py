# The file for training in CNN

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.5f}'.format

import model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# HyperParameters
IMG_SIZE = 227
INPUT_DEPTH = 24

BATCH_SIZE = 32

LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_COST = 0.01
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99


MODEL_PATH = 'model_depth_' + str(INPUT_DEPTH) + '/'

MODEL_NAME = 'CAE_model'

TRAINING_DATA_PATH = './training_data'
TRAINING_TYPE = '.npy'


def train(training_features):
    features = tf.placeholder(dtype = tf.float32, shape = [BATCH_SIZE, IMG_SIZE, IMG_SIZE, INPUT_DEPTH], name = 'features')
    targets = tf.placeholder(dtype = tf.float32, shape = [BATCH_SIZE, IMG_SIZE, IMG_SIZE, INPUT_DEPTH], name = 'targets')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_COST)
    
    pred = model.forward_propagation(features, regularizer)
    
    global_step = tf.Variable(0, trainable = False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # Moving average for all trainable variables
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    # The cost function -- norm 
    matrix_pow_2 = tf.pow(tf.subtract(targets, pred), 2)
    matrix_norm = tf.reduce_sum(matrix_pow_2, axis = [1,2,3])

    norm_cost = tf.reduce_mean(matrix_norm)/2
    
    # The regularization score
    reg_score = tf.add_n(tf.get_collection('reg_score'))
    tot_loss = norm_cost + reg_score
    
    # Number of videos used for training
    num_videos = training_features.shape[0]
    # Length of each video
    size_videos = np.array([v.shape[0] for v in training_features])
    
    # Number of frames for each video that can be used for training
    training_num = size_videos - BATCH_SIZE - INPUT_DEPTH + 1
    # Total
    tot_training_num = sum(training_num)
    decay_steps = tot_training_num//BATCH_SIZE
    
    # Exponential decay of learning rate
    learning_rate = tf.train.exponential_decay(learning_rate = LEARNING_RATE_BASE,
                                              global_step = global_step,
                                              decay_steps = decay_steps,
                                              decay_rate = LEARNING_RATE_DECAY,
                                              staircase = False)
    
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(tot_loss, global_step = global_step)
    train_op = tf.group(train_step, variables_averages_op)
    
    saver = tf.train.Saver(max_to_keep=6)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        train_losses = []
        
        #Train        
        cur_step = 0
        
        while(cur_step < TRAINING_STEPS):
            for cur_video_index in range(num_videos):
                first_flag = 1
                cur_training_features = training_features[cur_video_index]
                for i in range(training_num[cur_video_index]):
                    start = i
                    end = start + BATCH_SIZE
            
                    if(first_flag):
                        train_features = [np.transpose(cur_training_features[j:j+INPUT_DEPTH], axes = [1,2,0]) for j in range(BATCH_SIZE)]
                        first_flag = 0
                    else:
                        train_features.pop(0)
                        train_features.append(np.transpose(cur_training_features[end-1:end-1+INPUT_DEPTH], axes = [1,2,0]))

                    train_targets = train_features.copy()
                    train_feed = {features: train_features, targets:train_targets}

                    if(cur_step%50 == 0):
                        train_loss = sess.run(tot_loss, feed_dict = train_feed)
                        reg_loss = sess.run(reg_score, feed_dict = train_feed)

                        print("After %d training step(s), loss on training is: %g, regularization cost is:  %g" % (cur_step, train_loss, reg_loss))
                        print("The current video is: %d, the batch frame is from %d to %d" % (cur_video_index, start, end))

                        train_losses.append(train_loss)
                        
                    if(cur_step%100 == 0):
                        saver.save(sess, os.path.join(MODEL_PATH, MODEL_NAME), global_step=global_step)
                    #if(cur_step%1000 == 0):
                    sess.run(train_op, feed_dict = train_feed)
                    cur_step = cur_step + 1
                    if(cur_step >= TRAINING_STEPS):
                        break
                        
                if(cur_step >= TRAINING_STEPS):
                    break
    print(train_losses)


def main():
    dirs = os.listdir(TRAINING_DATA_PATH)
    dirs = [d for d in dirs if(TRAINING_TYPE in d)]

    training_features = [np.load(TRAINING_DATA_PATH + '/' +  t) for t in dirs]
    training_features = np.array(training_features)

    num_videos = training_features.shape[0]

    size_videos = [v.shape[0] for v in training_features]

    print(size_videos)

    training_features[2].shape[0]

    sum(size_videos)

    import importlib
    importlib.reload(model)

    tf.reset_default_graph()
    train(training_features)

if __name__ == '__main__':
    main()
