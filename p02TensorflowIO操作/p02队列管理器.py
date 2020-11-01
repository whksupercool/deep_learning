import tensorflow as tf
import os

# 消除警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 模拟异步子线程读取样本，存入样本，主线程读取样本
# 1. 定义一个队列，1000
Q = tf.FIFOQueue(1000, tf.float32)
# 2. 定义子线程要做的事情   循环的值+1，放入队列中
var = tf.Variable(0.0)
# 实现一个自增 tf.assign_add
data = tf.assign_add(var, tf.constant(1.0))
en_q = Q.enqueue(data)
# 3. 定义队列管理器op, 指定多少个子线程,以及子线程该做的事情
qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 2)
# 初始化变量的op
init_op = tf.global_variables_initializer()

# 开启会话
with tf.Session() as sess:
    # 初始化变量
    sess.run(init_op)
    # 开启线程管理
    coord = tf.train.Coordinator()
    # 真正开启子线程
    threads = qr.create_threads(sess, coord=coord, start=True)
    # 主线程,不断读取数据训练
    for i in range(300):
        print(sess.run(Q.dequeue()))

    # 回收
    coord.request_stop()
    coord.join(threads)
