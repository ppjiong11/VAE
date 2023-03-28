import numpy as np

# 读取.npy文件
data1 = np.load('cirs.npy') #一维时间序列
#data1 = np.load('generated_data.npy') #一维时间序列
print(data1)
# 获取数组形状
num_samples, input_dim = data1.shape #numpy中的shape方法获取数组形状，如3x3的矩阵就返回（3,3）。
#data.shape返回的是一个元组（tuple），包含了数组data的形状信息，例如(1000, 10)。
#在Python中，元组的元素可以通过序列解包（sequence unpacking）的方式直接赋值给多个变量。
#因此，可以使用num_samples, input_dim = data.shape的语法将元组中的两个元素（即样本数和输入维度）分别赋值给变量num_samples和input_dim。

# 打印数组形状
print('样本数:', num_samples)
print('输入维度:', input_dim)
data_npy = data1.reshape(num_samples, input_dim)
print(data_npy)