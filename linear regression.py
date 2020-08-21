#画出RM,DIS,PTRATIO,LSTAT与y的散点图,分析特征与y是否有线性关系?
#尝试进行线性回归,使用RM,DIS,PTRATIO,LSTAT预测房价y,写出回归方程
#解释下RM与Y的关系?
#对某新小区,其RM=8,DIS=2,PTRATIO=12,LSTAT=22,预测该小区房价
#获取数据
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x_df=pd.DataFrame(datasets.load_boston()['data'],columns=datasets.load_boston()['feature_names']) #X
y_df=pd.DataFrame(datasets.load_boston()['target'],columns=['y']) #Y
df=x_df.join(y_df)

#散点图
RM = df['RM']
DIS = df['DIS']
PTRATIO = df['PTRATIO']
LSTAT = df['LSTAT']
y = df['y']

plt.subplot(4,1,1)
plt.scatter(RM,y)
plt.subplot(4,1,2)
plt.scatter(DIS,y)
plt.subplot(4,1,3)
plt.scatter(PTRATIO,y)
plt.subplot(4,1,4)
plt.scatter(LSTAT,y)
plt.show()
###  通过绘制的散点图可以发现房价y与房间数RM呈现正相关，y随着RM的增加而增长；
###  房价y与距离就业中心的加权距离DIS无明显相关关系；
###  房价y与城镇中教师学生比例PTRATIO无明显相关关系；
###  房价y与房东属于低收入人群LSTAT呈现负相关，y随着LSTAT的增加而降低。

#线性回归
data = df[['RM','DIS','PTRATIO','LSTAT']]
lineR = LinearRegression()
lineR.fit(data,y)
w = lineR.coef_            #print(w)
b = lineR.intercept_       #print(b)
print('回归方程为：','\n','y=',w[0],'*RM+',w[1],'*DIS+',w[2],'*PTRATIO+',w[3],'*LSTAT+',b)
###  通过该回归方程可以发现，当其他变量(DIS,PTRATIO,LSTAT)不变时，
###  RM增加一个单位，房价平均增加4.2238个单位；RM减少一个单位，房价平均降低4.2238个单位


#预测
x = np.array([8,2,12,22]).reshape(1, -1)
y_pre = lineR.predict(x)
print('当RM=8,DIS=2,PTRATIO=12,LSTAT=22时预测该小区房价为：',y_pre)


###########OLS方法(包含各种检验结果）
import statsmodels.api as sm
yy = df['y']
xx = df[['RM','DIS','PTRATIO','LSTAT']]
XX = sm.add_constant(xx)#给自变量中加入常数项
model = sm.OLS(yy,XX)
result = model.fit()
print(result.summary())



