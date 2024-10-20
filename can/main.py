import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 
import os
from absl import app
from absl import flags
from cgan import CGAN

flags = tf.app.flags

flags.DEFINE_integer("epochs",1000,"epochs per trainingstep")
flags.DEFINE_float("learning_rate",0.0001,"learning rate for the model")
flags.DEFINE_integer("image_size",128,"Image size of the input")
flags.DEFINE_bool("training",True,"running training of the poincloud gan")
flags.DEFINE_string("checkpoint_dir","can/checkpoint","where to save the model")
flags.DEFINE_integer("iterations",100000,"number of patches")
flags.DEFINE_integer("batch_size",64,"size of the batch")
flags.DEFINE_float("beta1",0.5,"adam beta1")
flags.DEFINE_float("beta2",0.9,"adam beta2")
flags.DEFINE_integer("z_dim",128,"sample size from the normal distribution for the generator")
FLAGS = flags.FLAGS

def _main(argv):
	print("initializing Params")
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	if not os.path.exists("can/new_data"):
		os.makedirs("can/new_data")

	cgan = CGAN(FLAGS.training,FLAGS.epochs,FLAGS.checkpoint_dir,FLAGS.learning_rate,FLAGS.z_dim,FLAGS.batch_size,FLAGS.beta1,FLAGS.beta2,FLAGS.image_size)
	cgan.train()

if __name__ == '__main__':
	app.run(_main)
