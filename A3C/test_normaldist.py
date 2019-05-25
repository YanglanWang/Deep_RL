import tensorflow as tf
mu=tf.convert_to_tensor([0.],dtype=tf.float32)
sigma=tf.convert_to_tensor([1.],dtype=tf.float32)
normal_dist=tf.distributions.Normal(mu,sigma)
a=tf.squeeze(normal_dist.sample(1),axis=[0,1])
b=tf.convert_to_tensor([1,2,3],dtype=tf.int32)
c=tf.convert_to_tensor([11,21,31],dtype=tf.int32)
d=[tf.assign(c_tmp,b_tmp) for c_tmp,b_tmp in zip(b,c)]
with tf.Session() as sess:
    print(sess.run(a))
    sess.run(d)
    print(sess.run(c))