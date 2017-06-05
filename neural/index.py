# coding: utf-8
import sys
import os
import tensorflow as tf
from PIL import Image
from six.moves import cPickle as Pickle

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(root_dir)
import dataset
import inference

IMAGE_HEIGHT = 45
IMAGE_WIDTH = 45
IMAGE_CHANNEL = 1
LEARNING_RATE_BASE = 1e-2
LEARNING_RATE_DECAY = 0.99
MOVING_AVERAGE = 0.99  # 平均滑动
MAX_STEPS = 40000
ROTATE_ANGLE = 15
ROTATE_COUNTS = 6
BATCH_SIZE = 1000

# 路径设置
datasets_dir = os.path.join(root_dir, "datasets")
models_dir = os.path.join(root_dir, "models")
models_file = os.path.join(models_dir, "model.ckpt")

image_holder = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
label_holder = tf.placeholder(tf.int32, [None])


def train():
	"""
	
	:return: 
	"""
	# 读取数据集
	filenames = os.listdir(datasets_dir)
	# 过滤不合格数据集
	for filename in filenames:
		if not os.path.splitext(filename)[1] == '.pickle':
			filenames.remove(filename)

	logits = inference.inference(image_holder, reuse=False)
	global_step = tf.Variable(0, trainable=False)
	# 定义滑动平滑平均值
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE, global_step)
	variable_averages_op = variable_averages.apply(tf.trainable_variables())
	# 损失函数值
	loss = inference.loss(logits, label_holder)
	# 使用反向传播函数之前优化学习率
	learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, MAX_STEPS,
											   decay_rate=LEARNING_RATE_DECAY)
	# 定义反向传播函数
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	# 使用反向函数和滑动平滑值更新参数
	train_op = tf.group(train_step, variable_averages_op)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		tf.train.start_queue_runners()
		if not os.path.exists(models_dir):
			os.makedirs(models_dir)
		for step in range(MAX_STEPS):
			for filename in filenames:
				train_image, train_label = dataset.read(filename)
				assert isinstance(train_image, list)
				assert isinstance(train_label, list)
				_, loss_value = sess.run([train_op, loss], feed_dict={image_holder: train_image, label_holder: train_label})
			if step % 100 == 0:
				print("after %d steps, the loss value is %g" % (step, loss_value))
				saver.save(sess, models_file, global_step=step)


def img_rotate(img_dir, file):
	"""
	
	:param img_dir: 
	:param file: 
	:return: 
	"""
	img = Image.open(os.path.join(datasets_dir, file))
	image_list = []
	for rotate_index in range(ROTATE_COUNTS):
		img = img.rotate(rotate_index * ROTATE_ANGLE)
		image_list.append(img)


if __name__ == "__main__":
	train()
