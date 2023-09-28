import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

A = tf.constant([[4, 3], [6, 1]])
B = tf.constant([[1, 2], [3, 4]])

tensor_zero = tf.zeros(shape=[3, 4], dtype=tf.float32)
tensor_one = tf.ones(shape=[3, 4], dtype=tf.float32)
reshape_tensor = tf.reshape(tensor=tensor_one, shape=[4, 3])
AB_concat = tf.concat(values=[A, B], axis=1)

tensor_cast = tf.cast(reshape_tensor, tf.int32)

tensor_transpose = tf.transpose(AB_concat)

tensor_matmul = tf.matmul(AB_concat, tensor_transpose)

tensor_identity = tf.eye(3, 3, dtype=tf.int32)

print(tensor_identity)
# print(AB_concat)
