# 1近河住宅和非近河住宅的房价上是否有差异?(提示:探索下CHAS与y的关系,即按照CHAS分组汇总y信息)
# 2假定两种类型房屋的房价服从正态分布,使用T检验探查近河住宅和非近河住宅在房价上的差异是否有统计学意义?(显著性水平为0.05)


import pandas as pd
from sklearn import datasets
x_df=pd.DataFrame(datasets.load_boston()['data'],columns=datasets.load_boston()['feature_names']) #X
y_df=pd.DataFrame(datasets.load_boston()['target'],columns=['y']) #Y
df=x_df.join(y_df)

#1
data_y = df['y'][df['CHAS']==1]
data_n = df['y'][df['CHAS']==0]
print(data_y.describe())
print(data_n.describe())

#2
from scipy import stats
stats.levene(data_y,data_n)    #检验方差齐性:如果返回结果的p值<0.05，认为两总体不具有方差齐性
[T,p]=stats.ttest_ind(data_y,data_n,equal_var=False) 
print('p=',p)
#p=0.0036<0.05 拒绝原假设，认为两种类型房屋的房价在置信水平95%下有显著差异。