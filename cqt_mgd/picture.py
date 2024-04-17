#可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#读取csv中指定列的数据

#创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
#df = pd.DataFrame(columns=['epoch','train Loss','training accuracy','dev acc'])
plt.switch_backend('agg')
path='train.csv'


data = pd.read_csv(path)
epoch=data[['epoch']]
data_loss = data[['train Loss']] #class 'pandas.core.frame.DataFrame'
data_acc = data[['training accuracy']]
dev_acc=data[['dev acc']]
x = np.array(epoch)
y1 =np.array(data_loss)#将DataFrame类型转化为numpy数组
y2 = np.array(data_acc)
y3=np.array(dev_acc)
#绘图
plt.plot(x,y1,label="loss")
plt.plot(x,y2,label="train_accuracy")
plt.title("loss & accuracy") 
plt.xlabel('step')
plt.ylabel('probability')
plt.legend()   #显示标签
plt.show()
plt.savefig("./pic.jpg")