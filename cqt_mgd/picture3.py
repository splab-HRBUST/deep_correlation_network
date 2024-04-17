import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

 
#输入因变量
y1=[0.1657,0.1552,0.1633,0.1771]
y2=[0.1531,0.1422,0.1598,0.1692]
y3=[0.1831,0.1489,0.1789,0.1871]

fig,ax=plt.subplots(figsize=(6.4,4.8), dpi=100)
#x = ["BerNB", "MultiNB", "LogReg", "SVC" ,"LSVC", "NuSVC"]
#设置自变量的范围和个数
x = [" 0.01", " 0.001", " 0.0001", " 0.00001"]
#画图
ax.plot(x,y1, label='s1s3', linestyle='-', marker='*',  markersize='10')
ax.plot(x,y2, label='s1s2', linestyle='-.', marker='o', markersize='10')
ax.plot(x,y3, label='s2s3', linestyle='--', marker='p', markersize='10')
#设置坐标轴
#ax.set_xlim(0, 9.5)
#ax.set_ylim(0, 1.4)
ax.set_xlabel('λ ', fontsize=13)
ax.set_ylabel('t-dcf(×100)', fontsize=13)
#设置刻度
ax.tick_params(axis='both', labelsize=11)
#显示网格
#ax.grid(True, linestyle='-.')
ax.yaxis.grid(True, linestyle='-.')
#添加图例
legend = ax.legend(loc='best')
 
plt.show()
fig.savefig('./t-dcf.png')