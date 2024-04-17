import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import preprocessing
 
a = np.load("resnet.npy") 
print(a)  
print("数组元素总数：",a.size)      #打印数组尺寸，即数组元素总数  

print("数组形状：",a.shape)         #打印数组形状

# row_rand_array = np.arange(a.shape[0])#shape[0]是有几行,shape[1]是有几列
# print(row_rand_array)#[0 1 2] 相当于行的index 表示有3行

# np.random.shuffle(row_rand_array)#将3行对应的3个index [0 1 2] 打乱
# row_rand = a[row_rand_array[0:2000]]#3个index打乱后取前2个,也就相当于matrix行数打乱后取前2行
# print(row_rand)#可能长这样：[[5  6  7  8  9],[0  1  2  3  4]]，因为随机所以每次都是不一样的2行


 