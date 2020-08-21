#忽略数据标准化,使用Kmeans聚类,设定k值为3,输出所有样本点的聚类结果,3个类别的类均值
#尝试以花萼长度(sepal length),花瓣长度(petal length)为x,y轴,可视化聚类结果

#导入数据
import pandas as pd
from sklearn import datasets
df=pd.DataFrame(datasets.load_iris()['data'],columns=datasets.load_iris()['feature_names']) 

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#聚类结果
clf = KMeans(n_clusters=3)
clf.fit(df)
centers = clf.cluster_centers_ # 两组数据点的中心点
labels  = clf.labels_          # 每个数据点所属分组
print('3个类别的类均值：','\n',centers)

result = df.copy()
result['label'] =pd.DataFrame(labels)
print('所有样本点的聚类结果：','\n',result)

#结果可视化
for i in range(len(labels)):
    if labels[i] == 0:
       c='c'
    elif labels[i] == 1:
       c='g'
    else:
       c='y'
    plt.scatter(df.ix[i][0], df.ix[i][2], c=c) 
plt.scatter(centers[:,0], centers[:,2], marker='*',c='k')




