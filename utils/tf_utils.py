import numpy as np
import tensorflow as tf

def get_vars(scope_name):
	vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
	assert len(vars) > 0
	return vars

def get_reg_loss(scope_name):
	return tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope_name))
