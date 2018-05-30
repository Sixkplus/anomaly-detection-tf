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

import model_test
import os
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# HyperParameters
IMG_SIZE = 227
#INPUT_DEPTH = 20
from model_test import INPUT_DEPTH as INPUT_DEPTH

BATCH_SIZE = 1

LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_COST = 0.01
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99


MODEL_PATH = 'model_depth_' + str(INPUT_DEPTH) + '/'

MODEL_NAME = 'CAE_model'

TEST_DATA_PATH = './testing_data'
TEST_TYPE = '.npy'

OUT_IMAGE_PATH = './spatial_loss_images/'

def test(test_features):
    features = tf.placeholder(dtype = tf.float32, shape = [BATCH_SIZE, IMG_SIZE, IMG_SIZE, INPUT_DEPTH], name = 'features')
    targets = tf.placeholder(dtype = tf.float32, shape = [BATCH_SIZE, IMG_SIZE, IMG_SIZE, INPUT_DEPTH], name = 'targets')
    
    pred = model_test.forward_propagation(features, None)
    
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    # Moving average for all trainable variables
    variables_to_restore = variable_averages.variables_to_restore()
    
    # The cost function -- norm 
    matrix_pow_2 = tf.pow(tf.subtract(targets, pred), 2)
    matrix_norm = tf.reduce_sum(matrix_pow_2, axis = [1,2,3])

    norm_cost = tf.reduce_mean(matrix_norm)/2
    
    # Number of videos used for testing
    num_videos = test_features.shape[0]
    # Length of each video
    size_videos = np.array([v.shape[0] for v in test_features])
    
    # Number of frames for each video that can be used for test
    test_num = size_videos - BATCH_SIZE - INPUT_DEPTH + 1
    
    saver = tf.train.Saver(variables_to_restore)
    
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        
        if (ckpt and ckpt.all_model_checkpoint_paths):
            for model_path in ckpt.all_model_checkpoint_paths:
                print(model_path)
                saver.restore(sess, model_path)
                global_step = model_path.split('-')[-1]
                
                # The regularity score for each video
                E = []
                flag = 1
                
                for cur_video_index in range(num_videos):
                    first_flag = 1
                    cur_test_features = test_features[cur_video_index]
                    e = []
                    print("current video: %d" %cur_video_index)
                    for i in range(0,test_num[cur_video_index],1):
                        start = i
                        end = start + BATCH_SIZE
                        #print("current frame: %d" %i)
                        if(first_flag):
                            testing_features = [np.transpose(cur_test_features[j:j+INPUT_DEPTH], axes = [1,2,0]) for j in range(BATCH_SIZE)]
                            first_flag = 0
                        else:
                            testing_features.pop(0)
                            testing_features.append(np.transpose(cur_test_features[end-1:end-1+INPUT_DEPTH], axes = [1,2,0]))

                        test_targets = testing_features.copy()
                        test_feed = {features: testing_features, targets:test_targets}
                        
                        test_loss = sess.run(norm_cost, feed_dict = test_feed)
                        
                        # [1, 227, 227, depth]
                        image_loss = sess.run(matrix_pow_2, feed_dict = test_feed)
                        
                        # [depth, 227, 227]
                        image_loss = np.transpose(image_loss[0], axes = [2, 0, 1])
                        
                        # [227,227]
                        image_loss = image_loss[0]
                        
                        image_loss = cv2.resize(image_loss, (150,113), interpolation = cv2.INTER_AREA)
                        
                        # [3, 227, 227]
                        #image_loss = cv2.merge([image_loss,image_loss,image_loss])

                        
                        #print(image_loss.shape)
                        
                        print(i, cv2.imwrite(OUT_IMAGE_PATH + str(i) + '.jpg',image_loss*255))
                        
                        e.append(test_loss)
                        
                    print(np.min(e), np.max(e))
                    if(flag):
                        min_e = np.min(e)
                        max_e = np.max(e)
                        flag = 0
                    else:
                        min_e = min(min_e, np.min(e))
                        max_e = max(max_e, np.max(e))
                        
                    E.append(np.array(e))
                    
                np.save('rawScore' + str(INPUT_DEPTH) + model_path.split(MODEL_PATH)[1] + '.npy',E)
                
                s = [np.clip(np.array(1 - (e - np.min(e))/(np.max(e) - np.min(e))), 0.0, 1.0) for e in E]
                s1 = np.array(s)
                np.save('res_' + str(INPUT_DEPTH)+ model_path.split(MODEL_PATH)[1] +'.npy',s1)
                    
                s = [np.clip(np.array(1 - (e - min_e)/(max_e - min_e)), 0.0, 1.0) for e in E]
                s1 = np.array(s)
                np.save('resGlobal_' + str(INPUT_DEPTH)+ model_path.split(MODEL_PATH)[1] +'.npy',s1)    
            return E, min_e, max_e
        else:
            print("No valide model found, please check the model path and file name")
            return None


def main():
    dirs = os.listdir(TEST_DATA_PATH)
    dirs = [d for d in dirs if('.npy' in d and 'test' in d)]
    dirs.sort(key = lambda x:int((x.split('test')[1]).split('.npy')[0]))
    print(dirs)
    testing_features = [np.load(TEST_DATA_PATH + '/' + t) for t in dirs]
    testing_features = np.array(testing_features)
    
    tf.reset_default_graph()
    test(testing_features)
    '''
    E, min_e, max_e = test(testing_features)
    np.save('rawScore' + str(INPUT_DEPTH) +'.npy',E)
    s = [np.clip(np.array(1 - (e - min_e)/(max_e - min_e)), 0.0, 1.0) for e in E]
    s1 = np.array(s)
    np.save('global_minmax_res_' + str(INPUT_DEPTH) +'.npy',s1)
    '''

if __name__ == '__main__':
    main()