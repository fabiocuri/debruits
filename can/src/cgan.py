import os
import time

import numpy as np
import tensorflow as tf

from util import load_data_art, save_image, shuffle_data


class CGAN(object):

	def __init__(self, flags):
		""""
		Args:
			beta1: beta1 for AdamOptimizer
			beta2: beta2 for AdamOptimizer
			learning_rate: learning_rate for the AdamOptimizer
			training: [bool] Training/NoTraining
			batch_size: size of the batch_
			epoch: number of epochs
			checkpoint_dir: directory in which the model will be saved
			name_art: name of the fake_art will be saved
			image_size: size of the image
			z_dim: sample size from the normal distribution for the generator
		"""

		self.FLAGS = flags

		self.beta1 = self.FLAGS.beta1
		self.beta2 = self.FLAGS.beta2
		self.learning_rate = self.FLAGS.learning_rate
		self.training = self.FLAGS.training
		self.batch_size = self.FLAGS.batch_size
		self.epoch = self.FLAGS.epochs
		self.checkpoint_dir = self.FLAGS.checkpoint_dir
		self.name_art = "fake_art"
		self.image_size = self.FLAGS.image_size
		self.z_dim = self.FLAGS.z_dim
		self.scaler = 10.0
		self.save_epoch = 0
		self.build_network()

	def generator(self,x):
		with tf.variable_scope("generator") as scope:

			#x = tf.layers.dense(x,1024,activation=None,name="gan_input_layer")
			#x = tf.layers.batch_normalization(x,momentum=0.9,gamma_initializer = tf.random_normal_initializer(1., 0.02))
			#x = tf.nn.leaky_relu(x)
			x = tf.reshape(x, [-1, 1, 1,128])
			#size 5 output_size = strides * (input_size-1) + kernel_size - 2*padding padding = valid padding = 0
			x = tf.layers.conv2d_transpose(x,filters=1024,kernel_size=5, kernel_initializer=tf.glorot_normal_initializer(),strides=2,padding='valid',name="gan_deconv_1")
			x = tf.layers.batch_normalization(x,momentum=0.9,gamma_initializer = tf.random_normal_initializer(1., 0.02))
			x = tf.nn.leaky_relu(x)
			#size 13
			x = tf.layers.conv2d_transpose(x,filters=512,kernel_size=5,kernel_initializer=tf.glorot_normal_initializer(),strides=2,padding='valid',name="gan_deconv_2")
			x = tf.layers.batch_normalization(x,momentum=0.9,gamma_initializer = tf.random_normal_initializer(1., 0.02))
			x = tf.nn.leaky_relu(x)
			#size 29
			x = tf.layers.conv2d_transpose(x,filters=256,kernel_size=5,kernel_initializer=tf.glorot_normal_initializer(),strides=2,padding='valid',name="gan_deconv_3")
			x = tf.layers.batch_normalization(x,momentum=0.9,gamma_initializer = tf.random_normal_initializer(1., 0.02))
			x = tf.nn.leaky_relu(x)
			#size 61
			x = tf.layers.conv2d_transpose(x,filters=128,kernel_size=5,kernel_initializer=tf.glorot_normal_initializer(),strides=2,padding='valid',name="gan_deconv_4")
			x = tf.layers.batch_normalization(x,momentum=0.9,gamma_initializer = tf.random_normal_initializer(1., 0.02))
			x = tf.nn.leaky_relu(x)
			#size 125
			x = tf.layers.conv2d_transpose(x,filters=3,kernel_size=8,kernel_initializer=tf.glorot_normal_initializer(),strides=2,padding='valid',name="gan_deconv_5")
			x = tf.nn.tanh(x)
			#x = tf.reshape(x,[self.batch_size,self.image_size,self.image_size,3])
		return x


	def discriminator(self,x,reuse=False):
		with tf.variable_scope("discriminator" ,reuse=reuse):

			x = tf.layers.conv2d(x,filters=128,kernel_size=5,kernel_initializer=tf.glorot_normal_initializer(),strides=(2,2),padding='same',activation = None,name="conv_1")
			x = tf.layers.batch_normalization(x,momentum=0.9,gamma_initializer = tf.random_normal_initializer(1., 0.02))
			x = tf.nn.leaky_relu(x)

			x = tf.layers.conv2d(x,filters=256,kernel_size=5,activation = None,kernel_initializer=tf.glorot_normal_initializer(),strides=2,padding='same',name="conv_3")
			x = tf.layers.batch_normalization(x,momentum=0.9,gamma_initializer = tf.random_normal_initializer(1., 0.02))
			x = tf.nn.leaky_relu(x)

			x = tf.layers.conv2d(x,filters=512,kernel_size=5,activation = None,kernel_initializer=tf.glorot_normal_initializer(),strides=2,padding='same',name="conv_4")
			x = tf.layers.batch_normalization(x,momentum=0.9,gamma_initializer = tf.random_normal_initializer(1., 0.02))
			x = tf.nn.leaky_relu(x)

			x = tf.layers.conv2d(x,filters=1024,kernel_size=5,activation = None,kernel_initializer=tf.glorot_normal_initializer(),strides=2,padding='same',name="conv_5")
			x = tf.layers.batch_normalization(x,momentum=0.9,gamma_initializer = tf.random_normal_initializer(1., 0.02))
			x = tf.nn.leaky_relu(x)

			x = tf.layers.flatten(x)

			#Vanilla-GAN
			#x = tf.layers.dense(x,1,activation=tf.nn.sigmoid,kernel_initializer=tf.contrib.layers.xavier_initializer(),name="disc_output")

			#Wasserstein
			x = tf.layers.dense(x,1,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),name="disc_output")

		return x

	def build_network(self):

		self.input = tf.placeholder(tf.float32, [self.batch_size,self.image_size,self.image_size,3], name="real_art_picture")
		self.z = tf.placeholder(tf.float32,[None,self.z_dim], name ="noice")
		self.Gen = self.generator(self.z)
		self.Dis_real = self.discriminator(self.input,reuse = False)
		self.Dis_generator = self.discriminator(self.Gen,reuse = True)


		#Gradient penalty
		self.epsilon = tf.random_uniform([], 0.0, 1.0)
		self.x_hat = self.epsilon * self.input + (1 - self.epsilon) * self.Gen
		self.d_hat = self.discriminator(self.x_hat,reuse = True)
		self.ddx = tf.gradients(self.d_hat, self.x_hat)[0]
		self.ddx = tf.sqrt(tf.reduce_sum(tf.square(self.ddx), axis=1))
		self.ddx = tf.reduce_mean(tf.square(self.ddx - 1.0) * self.scaler)

		#Wassersteinmetrik
		self.d_loss = -tf.reduce_mean(self.Dis_real - self.Dis_generator)
		self.d_loss = self.d_loss + self.ddx
		self.g_loss = -tf.reduce_mean(self.Dis_generator)

		#Vanilla BI-GAN Loss
		#self.d_loss = -tf.reduce_mean(tf.log(self.Dis_real) + tf.log(1. - self.Dis_generator))
		#self.g_loss = -tf.reduce_mean(tf.log(self.Dis_generator))

		#Tensorboard variables
		self.d_sum_real = tf.summary.histogram("d_real", self.Dis_real)
		self.d_sum_fake = tf.summary.histogram("d_fake", self.Dis_generator)

		self.G_sum = tf.summary.histogram("G",self.Gen)
		self.z_sum = tf.summary.histogram("z_input",self.z)

		tf.summary.scalar('self.g_loss', self.g_loss )
		tf.summary.scalar('self.d_loss', self.d_loss )

		#collect generator and encoder variables
		self.vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
		self.vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

		print("init_d_optim")
		self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1 = self.beta1,beta2 = self.beta2).minimize(self.g_loss,var_list = self.vars_G)
		print("init_g_optim")
		self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1 = self.beta1,beta2 = self.beta2).minimize(self.d_loss,var_list = self.vars_D)

		self.saver = tf.train.Saver()

		#Tensorboard variables
		self.summary_g_loss = tf.summary.scalar("g_loss",self.g_loss)
		self.summary_d_loss = tf.summary.scalar("d_loss",self.d_loss)




	def save_model(self, iter_time):
		model_name = 'model'
		self.saver.save(self.sess, os.path.join(self.checkpoint_dir, model_name), global_step=iter_time)
		print('=====================================')
		print('             Model saved!            ')
		print('=====================================\n')


	def train(self):
		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth = True
		self.sess = tf.Session(config = self.config)

		with self.sess:
			if self.load_model():
				print(' [*] Load SUCCESS!\n')
			else:
				print(' [!] Load Failed...\n')
				self.sess.run(tf.global_variables_initializer())

			train_writer = tf.summary.FileWriter("./logs",self.sess.graph)
			merged = tf.summary.merge_all()
			self.counter = 1
			self.training_data = load_data_art()

			k = (len(self.training_data) // self.batch_size)
			self.start_time = time.time()
			loss_g_val,loss_d_val = 0, 0
			self.training_data = self.training_data[0:(self.batch_size*k)]

			for e in range(0,self.epoch):
				epoch_loss_d = 0.
				epoch_loss_g = 0.
				self.training_data = shuffle_data(self.training_data)
				for i in range(0,k):
					self.batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
					self.batch = self.training_data[i*self.batch_size:(i+1)*self.batch_size]
					self.batch = np.asarray(self.batch)
					_, loss_d_val,loss_d = self.sess.run([self.d_optim,self.d_loss,self.summary_d_loss],feed_dict={self.input: self.batch, self.z: self.batch_z})
					train_writer.add_summary(loss_d,self.counter)
					_, loss_g_val,loss_g = self.sess.run([self.g_optim,self.g_loss,self.summary_g_loss],feed_dict={self.z: self.batch_z, self.input: self.batch})
					train_writer.add_summary(loss_g,self.counter)
					self.counter=self.counter + 1
					epoch_loss_d += loss_d_val
					epoch_loss_g += loss_g_val
				epoch_loss_d /= k
				epoch_loss_g /= k
				print("Loss of D: %f" % epoch_loss_d)
				print("Loss of G: %f" % epoch_loss_g)
				print("Epoch%d" %(e))
				if e % 1 == 0:
					save_path = self.saver.save(self.sess,"checkpoint/model.ckpt",global_step=self.save_epoch)
					print("model saved: %s" %save_path)
					self.gen_noise = np.random.uniform(-1, 1, [1, self.z_dim])
					fake_art = self.sess.run([self.Gen], feed_dict={self.z: self.gen_noise})
					save_image(fake_art,self.name_art,self.save_epoch)
					self.save_epoch += 1
			print("training finished")

	def load_model(self):
		print(' [*] Reading checkpoint...')
		ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
			meta_graph_path = ckpt.model_checkpoint_path + '.meta'
			self.save_epoch = int(meta_graph_path.split('-')[-1].split('.')[0])
			print('===========================')
			print('   iter_time: {}'.format(self.save_epoch))
			print('===========================')
			return True
		else:
			return False
