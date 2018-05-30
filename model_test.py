# The file for Forward Propagation in NN

import tensorflow as tf
#from test import INPUT_DEPTH as INPUT_DEPTH

# Initialization and active function
#active_function = tf.nn.tanh

# The layer parameters

# Input information
IMG_SIZE = 227
INPUT_DEPTH = 20

# First layer -- convolution

CONV1_SIZE = 11
CONV1_DEPTH = 512
STRIDE1 = 4

# Second layer -- max-pooling
POOL1_SIZE = 2
STRIDE2 = 2

# Third layer -- convolution
CONV2_SIZE = 5
CONV2_DEPTH = 256
STRIDE3 = 1

# Forth layer -- max-pooling
POOL2_SIZE = 2
STRIDE4 = 2

# Fifth layer -- convolution
CONV3_SIZE = 3
CONV3_DEPTH = 128
STRIDE5 = 1

# 6th layer -- Deconvolution
DECONV1_SIZE = 3
DECONV1_DEPTH = 128
DECONV1_OUT_SIZE = 13
STRIDE6 = 1

# 7th layer -- unpooling
UNPOOL_SIZE1 = 2

DECONV2_SIZE = 3
DECONV2_DEPTH = 128
DECONV2_OUT_SIZE = 27

STRIDE7 = 2

# 8th layer -- Deconvolution
DECONV3_SIZE = 3
DECONV3_DEPTH = 256
DECONV3_OUT_SIZE = 27
STRIDE8 = 1

# 9th layer -- unpooling
UNPOOL_SIZE2 = 2

DECONV4_SIZE = 3
DECONV4_DEPTH = 256
DECONV4_OUT_SIZE = 55
STRIDE9 = 2

# 10th layer -- Deconvolution
DECONV5_SIZE = 5
DECONV5_DEPTH = 512
DECONV5_OUT_SIZE = 55
STRIDE10 = 1

# Output
DECONV6_SIZE = 11
DECONV6_DEPTH = INPUT_DEPTH
DECONV6_OUT_SIZE = 512
STRIDE11 = 4


# The function to get the variables and add to collection

def get_weight_variable(shape, regularizer):
  weights = tf.get_variable(name = "weights", shape = shape, initializer = tf.contrib.layers.xavier_initializer())
  # The variable will be add to count of regularization
  if regularizer != None:
    tf.add_to_collection('reg_score', regularizer(weights))
    
  return weights


def forward_propagation(input_tensor, regularizer):
    
    # Batch size
    batch_size = input_tensor.get_shape().as_list()[0]
    
    # The variables constructed in this scope will have tag 'Layer1'
    with tf.variable_scope("Layer1"):
        weights1 = get_weight_variable(shape = [CONV1_SIZE, CONV1_SIZE, INPUT_DEPTH, CONV1_DEPTH],regularizer = regularizer)
        biases1 = tf.get_variable(name = 'biases', shape = [CONV1_DEPTH], initializer = tf.constant_initializer(0.0))

        conv1_out = tf.nn.conv2d(input = input_tensor, 
                                  filter = weights1, 
                                  strides = [1,STRIDE1, STRIDE1, 1], 
                                  padding = "VALID")
        layer1_out = tf.nn.tanh(tf.nn.bias_add(value = conv1_out, bias = biases1))
    
    
    with tf.variable_scope("Layer2"):
        layer2_out = tf.nn.max_pool(value = layer1_out,
                                    ksize = [1,POOL1_SIZE, POOL1_SIZE, 1],
                                    strides = [1,STRIDE2, STRIDE2, 1], 
                                    padding = 'VALID')
    
    with tf.variable_scope("Layer3"):
        weights2 = get_weight_variable(shape = [CONV2_SIZE, CONV2_SIZE, CONV1_DEPTH, CONV2_DEPTH], regularizer = regularizer)
        biases2 = tf.get_variable(name = 'biases', shape = [CONV2_DEPTH], initializer = tf.constant_initializer(0.0))

        conv2_out = tf.nn.conv2d(input = layer2_out, 
                                  filter = weights2, 
                                  strides = [1,STRIDE3, STRIDE3, 1], 
                                  padding = 'SAME')
        layer3_out = tf.nn.tanh(tf.nn.bias_add(value = conv2_out, bias = biases2))
    
    with tf.variable_scope("Layer4"):
        layer4_out = tf.nn.max_pool(value = layer3_out,
                                    ksize = [1,POOL2_SIZE, POOL2_SIZE, 1],
                                    strides = [1,STRIDE4, STRIDE4, 1], 
                                    padding = 'VALID')
    
    with tf.variable_scope("Layer5"):
        weights3 = get_weight_variable(shape = [CONV3_SIZE, CONV3_SIZE, CONV2_DEPTH, CONV3_DEPTH], regularizer = regularizer)
        biases3 = tf.get_variable(name = 'biases', shape = [CONV3_DEPTH], initializer = tf.constant_initializer(0.0))

        conv3_out = tf.nn.conv2d(input = layer4_out, 
                                  filter = weights3, 
                                  strides = [1,STRIDE5, STRIDE5, 1], 
                                  padding = 'SAME')
        layer5_out = tf.nn.tanh(tf.nn.bias_add(value = conv3_out, bias = biases3))
    
    
    with tf.variable_scope("Layer6"):
        weights4 = get_weight_variable(shape = [DECONV1_SIZE, DECONV1_SIZE, DECONV1_DEPTH, CONV3_DEPTH], regularizer =regularizer)
        biases4 = tf.get_variable(name = 'biases', shape = [DECONV1_DEPTH], initializer = tf.constant_initializer(0.0))

        deconv1_out = tf.nn.conv2d_transpose(value = layer5_out, 
                                  filter = weights4,
                                  output_shape = [batch_size, 13, 13, DECONV1_DEPTH],
                                  strides = [1,STRIDE6, STRIDE6, 1], 
                                  padding = 'SAME')
        layer6_out = tf.nn.tanh(tf.nn.bias_add(value = deconv1_out, bias = biases4))
        # [batch, 13,13, 128]
    
    with tf.variable_scope("Layer7"):
        # Up pooling (use convolution to take place)
        weights5 = get_weight_variable(shape = [DECONV2_SIZE, DECONV2_SIZE, DECONV2_DEPTH, DECONV1_DEPTH], regularizer = regularizer)
        biases5 = tf.get_variable(name = 'biases', shape = [DECONV2_DEPTH], initializer = tf.constant_initializer(0.0))

        deconv2_out = tf.nn.conv2d_transpose(value = layer6_out, 
                                  filter = weights5,
                                  output_shape = [batch_size, 27, 27, DECONV2_DEPTH],
                                  strides = [1,STRIDE7, STRIDE7, 1], 
                                  padding = 'VALID')
        layer7_out = tf.nn.tanh(tf.nn.bias_add(value = deconv2_out, bias = biases5))
        # [batch, 27,27, 128]
    
    with tf.variable_scope("Layer8"):
        weights6 = get_weight_variable(shape = [DECONV3_SIZE, DECONV3_SIZE, DECONV3_DEPTH, DECONV2_DEPTH], regularizer=regularizer)
        biases6 = tf.get_variable(name = 'biases', shape = [DECONV3_DEPTH], initializer = tf.constant_initializer(0.0))

        deconv3_out = tf.nn.conv2d_transpose(value = layer7_out, 
                                  filter = weights6,
                                  output_shape = [batch_size, 27, 27, DECONV3_DEPTH],
                                  strides = [1,STRIDE8, STRIDE8, 1], 
                                  padding = 'SAME')
        layer8_out = tf.nn.tanh(tf.nn.bias_add(value = deconv3_out, bias = biases6))
        # [batch, 27,27, 256]
        
    with tf.variable_scope("Layer9"):
        # Up pooling (use convolution to take place)
        weights7 = get_weight_variable(shape = [DECONV4_SIZE, DECONV4_SIZE, DECONV4_DEPTH, DECONV3_DEPTH], regularizer=regularizer)
        biases7 = tf.get_variable(name = 'biases', shape = [DECONV4_DEPTH], initializer = tf.constant_initializer(0.0))

        deconv4_out = tf.nn.conv2d_transpose(value = layer8_out, 
                                  filter = weights7,
                                  output_shape = [batch_size, 55, 55, DECONV4_DEPTH],
                                  strides = [1,STRIDE9, STRIDE9, 1], 
                                  padding = 'VALID')
        layer9_out = tf.nn.tanh(tf.nn.bias_add(value = deconv4_out, bias = biases7))
        # [batch, 55,55, 256]
    
    with tf.variable_scope("Layer10"):
        weights8 = get_weight_variable(shape = [DECONV5_SIZE, DECONV5_SIZE, DECONV5_DEPTH, DECONV4_DEPTH], regularizer=regularizer)
        biases8 = tf.get_variable(name = 'biases', shape = [DECONV5_DEPTH], initializer = tf.constant_initializer(0.0))

        deconv5_out = tf.nn.conv2d_transpose(value = layer9_out, 
                                  filter = weights8,
                                  output_shape = [batch_size, 55, 55, DECONV5_DEPTH],
                                  strides = [1,STRIDE10, STRIDE10, 1], 
                                  padding = 'SAME')
        layer10_out = tf.nn.tanh(tf.nn.bias_add(value = deconv5_out, bias = biases8))
        # [batch, 55,55, 512]
    
    # The output layer
    with tf.variable_scope("Layer11"):
        weights9 = get_weight_variable(shape = [DECONV6_SIZE, DECONV6_SIZE, DECONV6_DEPTH, DECONV5_DEPTH], regularizer=regularizer)
        biases9 = tf.get_variable(name = 'biases', shape = [DECONV6_DEPTH], initializer = tf.constant_initializer(0.0))

        deconv6_out = tf.nn.conv2d_transpose(value = layer10_out, 
                                  filter = weights9,
                                  output_shape = [batch_size, 227, 227, DECONV6_DEPTH],
                                  strides = [1,STRIDE11, STRIDE11, 1], 
                                  padding = 'VALID')
        output = tf.nn.sigmoid(tf.nn.bias_add(value = deconv6_out, bias = biases9))
        # [batch, 227,227, 512]

    return output
  