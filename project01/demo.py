import tensorflow as tf
import os

# 消除警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 实现一个假发运算
a = tf.constant(5.0)
b = tf.constant(6.0)

sum1 = tf.add(a, b)

# 默认的这张图相当于是给程序分配一段内存
graph = tf.get_default_graph()
print(graph)

with tf.Session() as sess:
    print(sess.run(sum1))
    print(a.graph)
    print(sum1.graph)
    print(sess.graph)
