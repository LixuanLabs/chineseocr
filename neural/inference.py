# coding:utf-8
from __future__ import print_function
import tensorflow as tf

NUM_CHANNELS = 1

CONV1_SIZE = 3
CONV1_DEEP = 16

CONV2_SIZE = 3
CONV2_DEEP = 32

CONV3_SIZE = 3
CONV3_DEEP = 64

CONV4_SIZE = 3
CONV4_DEEP = 3

CONV5_SIZE = 5
CONV5_DEEP = 3

CONV6_SIZE = 5
CONV6_DEEP = 3


FC1_SIZE = 7000
FC2_SIZE = 4500
LOGIT_SIZE = 3752


# 定义生成参数函数
def variable_with_weight_loss(shape, stddev, wl):
	"""

	:param shape: 
	:param stddev: 
	:param wl: 
	:return: 
	"""
	var = tf.get_variable("weight", shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
	if wl is not None:
		weight_loss = tf.multiply(tf.nn.l2_loss(var), wl)
		tf.add_to_collection('losses', weight_loss)
	return var

# 计算
def inference(input_tensor, reuse):
	"""

	:param input_tensor: 
	:param reuse: 
	:return: 
	"""
	with tf.variable_scope("layers1-conv1", reuse=reuse):
		conv1_weights = variable_with_weight_loss(shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], stddev=5e-1, wl=0.0)
		conv1_biases = tf.get_variable("biases", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
		conv1 = tf.nn.conv2d(input_tensor, conv1_weights, [1, 1, 1, 1], padding='VALID')

		relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
	with tf.name_scope("layers2-pool1-lrn1"):
		pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
		lrn1 = tf.nn.local_response_normalization(pool1, bias=1.0, alpha=0.001/9.0, beta=0.75)

	with tf.variable_scope('layers3-conv2', reuse=reuse):
		conv2_weights = variable_with_weight_loss(shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], stddev=5e-1, wl=0.0)
		conv2_biases = tf.get_variable("biases", [CONV2_DEEP], initializer=tf.constant_initializer(0.1))
		conv2 = tf.nn.conv2d(lrn1, conv2_weights, [1, 1, 1, 1], padding='SAME')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
	with tf.name_scope("layers4-pool2-lrn2"):
		# 先LRN处理后ReLU处理
		lrn2 = tf.nn.local_response_normalization(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
		pool2 = tf.nn.max_pool(lrn2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	with tf.variable_scope('layers5-conv3', reuse=reuse):
		conv3_weights = variable_with_weight_loss(shape=[CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP], stddev=5e-1, wl=0.0)
		conv3_biases = tf.get_variable("biases", [CONV3_DEEP], initializer=tf.constant_initializer(0.1))
		conv3 = tf.nn.conv2d(pool2, conv3_weights, [1, 1, 1, 1], padding='SAME')
		relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
	with tf.name_scope("layers8-pool3-lrn3"):
		lrn3 = tf.nn.local_response_normalization(relu3, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
		pool3 = tf.nn.max_pool(lrn3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

	# 全连接层之前的预处理，将数据转为向量
	pool_shape = pool3.get_shape().as_list()
	nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
	print("node:", nodes)
	reshaped = tf.reshape(pool3, [-1, nodes])

	with tf.variable_scope("layer9-fc1", reuse=reuse):
		fc1_weights = variable_with_weight_loss(shape=[nodes, FC1_SIZE], stddev=0.04, wl=0.004)
		fc1_bias = tf.get_variable("biases", [FC1_SIZE], initializer=tf.constant_initializer(0.1))
		fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_bias)
	with tf.variable_scope("layer10-fc2", reuse=reuse):
		fc2_weights = variable_with_weight_loss(shape=[FC1_SIZE, FC2_SIZE], stddev=0.04, wl=0.004)
		fc2_bias = tf.get_variable("biases", [FC2_SIZE], initializer=tf.constant_initializer(0.1))
		fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_bias)
	with tf.variable_scope("layer11-logit", reuse=reuse):
		fc3_weights = variable_with_weight_loss(shape=[FC2_SIZE, LOGIT_SIZE], stddev=1/192.0, wl=0.0)
		fc3_bias = tf.get_variable("biases", [LOGIT_SIZE], initializer=tf.constant_initializer(0.0))
		logits = tf.add(tf.matmul(fc2, fc3_weights), fc3_bias)
	return logits

def loss(logits, labels):
	"""

	:param logits: 
	:param labels: 
	:return: 
	"""
	labels = tf.cast(labels, tf.int32)
	print("损失函数中的label", labels)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
	cross_entropy_mean = tf.reduce_mean(cross_entropy)

	tf.add_to_collection("losses", cross_entropy_mean)

	return tf.add_n(tf.get_collection('losses'))




