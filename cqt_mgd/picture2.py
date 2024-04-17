import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 输入因变量
y1 = [6.65, 5.96, 6.47, 8.09]
y2 = [5.58, 4.71, 6.35, 7.61]
y3 = [7.22, 5.61, 7.36, 7.93]

fig,ax=plt.subplots(figsize=(6.4,4.8), dpi=100)
# fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=100)

# 设置自变量的范围和个数
x = [" 0.01", " 0.001", " 0.0001", " 0.00001"]
# 画图

ax.plot(x,y1, label='s1s3', linestyle='-', marker='*',  markersize='10')
ax.plot(x,y2, label='s1s2', linestyle='-.', marker='o', markersize='10')
ax.plot(x,y3, label='s2s3', linestyle='--', marker='p', markersize='10')
# ax[0].plot(x, y1, label='s1s3', linestyle='-', marker='*', markersize='10')
# ax[0].plot(x, y2, label='s1s2', linestyle='-.', marker='o', markersize='10')
# ax[0].plot(x, y3, label='s2s3', linestyle='--', marker='p', markersize='10')
# 设置坐标轴

# ax[0].set_xlabel('λ ', fontsize=13)
# ax[0].set_ylabel('EER(%)', fontsize=13)
# 设置刻度
# ax[0].tick_params(axis='both', labelsize=11)
# 显示网格
# ax.grid(True, linestyle='-.')
# ax[0].yaxis.grid(True, linestyle='-.')
# 添加图例
# legend = ax[0].legend(loc='best')
# ax[0].set_title('(a)', y=-2)
# y4 = [0.1657, 0.1552, 0.1633, 0.1771]
# y5 = [0.1531, 0.1422, 0.1598, 0.1692]
# y6 = [0.1831, 0.1489, 0.1789, 0.1871]

# fig,ax=plt.subplots(figsize=(6.4,4.8), dpi=100)

# 设置自变量的范围和个数
# x2 = [" 0.01", " 0.001", " 0.0001", " 0.00001"]
# # 画图
# ax[1].plot(x2, y4, label='s1s3', linestyle='-', marker='*', markersize='10')
# ax[1].plot(x2, y5, label='s1s2', linestyle='-.', marker='o', markersize='10')
# ax[1].plot(x2, y6, label='s2s3', linestyle='--', marker='p', markersize='10')
# # 设置坐标轴

# ax[1].set_xlabel('λ ', fontsize=13)
# ax[1].set_ylabel('t-DCF(%)', fontsize=13)
# # 设置刻度
# ax[1].tick_params(axis='both', labelsize=11)
# # 显示网格
# # ax.grid(True, linestyle='-.')
# ax[1].yaxis.grid(True, linestyle='-.')
# ax[0].set_title('(b)', y=-4)


ax.set_xlabel('λ ', fontsize=13)
ax.set_ylabel('EER(%)', fontsize=13)
#设置刻度
ax.tick_params(axis='both', labelsize=11)
#显示网格
#ax.grid(True, linestyle='-.')
ax.yaxis.grid(True, linestyle='-.')
#添加图例
legend = ax.legend(loc='best')
 
plt.show()
fig.savefig('./eer.png')
