import tensorflow as tf
import os

# 消除警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建一个图，包含一组op和tensor，上下文环境
# op:只要使用tensorflow的API定义的函数都是OP
# tensor(张量)：就指代的是数据
g = tf.Graph()
print(g)
with g.as_default():
    c = tf.constant(11.0)
    print(c.graph)

# 实现一个假发运算
a = tf.constant(5.0)
b = tf.constant(6.0)

sum1 = tf.add(a, b)

# 默认的这张图相当于是给程序分配一段内存
graph = tf.get_default_graph()
print(graph)

# 只能运行一个图结构,可以在会话中利用“graph=”参数指定图去运行
# 只要有上下文环境就能用eval（）

# 训练模型
# 实时的提供数据去进行训练

# placeholder是一个占位符,feed_dict一个字典
plt = tf.placeholder(tf.float32, [None, 3])
print(plt)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(plt, feed_dict={plt: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}))
    print(sum1.eval())
    print(a.graph)
    print(sum1.graph)
    print(sess.graph)
