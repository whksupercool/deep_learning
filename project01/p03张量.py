import tensorflow as tf
import os

# 消除警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# a = tf.constant(5.0)
# b = tf.constant(6.0)
# sum1 = tf.add(a, b)
#
# # placeholder是一个占位符,feed_dict一个字典
# plt = tf.placeholder(tf.float32, [None, 3])
# print(plt)
# print("==" * 50)
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#     # print(sum1.eval())
#     print("a.graph", "==" * 50)
#     print(a.graph)
#     print("a.shape", "==" * 50)
#     print(a.shape)
#     print("plt.shape", "==" * 50)
#     print(plt.shape)
#     print("a.name", "==" * 50)
#     print(a.name)
#     print("a.op", "==" * 50)
#     print(a.op)

# tensorflow:打印出来的形状表示
# 0维：()   1维:(5)  2维：(5,6)   3维：(2,3,4)

# 形状的概念
# 静态形状和动态性状
# 对于静态形状来说，一旦张量形状固定了，不能再次设置静态形状, 不能夸维度修改 1D->1D 2D->2D
# 动态形状可以去创建一个新的张量,改变时候一定要注意元素数量要匹配  1D->2D  1->3D
plt = tf.placeholder(tf.float32, [None, 2])
print(plt)

plt.set_shape([3, 2])
print(plt)

plt_reshape = tf.reshape(plt, [2, 3])
print(plt_reshape)

with tf.Session() as sess:
    pass
