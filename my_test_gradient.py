import tensorflow as tf
import numpy as np
# q=tf.convert_to_tensor(10.,dtype=tf.float32)
a=tf.convert_to_tensor([[1.,2.,3.]],dtype=tf.float32)
b=tf.convert_to_tensor(np.vstack([1.,2.,3.]),dtype=tf.float32)
c=tf.convert_to_tensor([[5.]],dtype=tf.float32)
q=tf.matmul(a,b)
a_grads = tf.gradients(q, a)
a_grads1 = tf.gradients(q, a,grad_ys=c)
with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(q))
    print(sess.run(a_grads))
    print(sess.run(a_grads1))