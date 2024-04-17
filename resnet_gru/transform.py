import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import preprocessing
#import pylab
plt.switch_backend('agg')
file = "gru.npy"
a = np.load(file)
# row_rand_array = np.arange(a.shape[0])#shape[0]是有几行,shape[1]是有几列
# # print(row_rand_array)#[0 1 2] 相当于行的index 表示有3行

# np.random.shuffle(row_rand_array)#将3行对应的3个index [0 1 2] 打乱
# row_rand = a[row_rand_array[0:4000]]#3个index打乱后取前2个,也就相当于matrix行数打乱后取前2行

zeros = np.zeros(2000)
ones = np.ones(2000)
labels = np.concatenate([zeros,ones],axis=0)
X_embedded = TSNE(perplexity=10,n_components=2,init="pca",early_exaggeration=100).fit_transform(a)
print(X_embedded.shape)
print(X_embedded)
result = X_embedded
# 颜色设置
color = ['#FFFAFA', '#BEBEBE', '#000080', '#87CEEB', '#006400',
         '#00FF00', '#4682B4', '#D02090', '#8B7765', '#B03060']
# 可视化展示
plt.figure(figsize=(10, 10))
plt.xticks([])
plt.yticks([])  
# plt.title('α=1.5579 and CQT pass BatchNorm')
plt.scatter(result[:,0], result[:,1], c=labels, s=10)
#plt.scatter(result[:,0], result[:,1], c=np.squeeze(result[:,1]), s=10)

# for i in range(2000): #说明黑色是欺诈语音，黄色是真实语音
#     print(labels[i])
#     plt.scatter(result[i,0], result[i,1], c=labels[i], s=10)

plt.show()
plt.savefig("gru1.jpg")
#pylab.show()