#通过花的属性数据预测花的种类,尝试构建决策树模型,要求最大树深度为3
#尝试预测sepal length=6,sepal width=1,petal length=3,petal width=1最可能是什么类型的花
#可视化决策树,解释下什么样的花最可能是setosa

#导入数据
import numpy as np
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split

X,y=datasets.load_iris(return_X_y=True) #X与y
target_names=datasets.load_iris().target_names #y的值列表:0:setosa,1:versicolor,2:virginica
feature_names=datasets.load_iris().feature_names #特征X的名称列表

# 构建决策树模型分割数据集
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)#分割数据集,训练集比例                                                   
print("\n训练集样本大小：", x_train.shape)
print("训练集标签大小：", y_train.shape)
print("测试集样本大小：", x_test.shape)
print("测试集标签大小：", y_test.shape)
clf = tree.DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=1)#设置决策树分类器,默认criterion="gini",基尼系数
clf.fit(x_train, y_train)  # 训练模型
score = clf.score(x_test, y_test) # 评价模型
print("模型测试集准确率为：", score)

#  预测  
a = np.array([[6.,1.,3.,1.]])
b = clf.predict(a)
clf.predict_proba(a)
if int(b) == 0:
    c='setosa'
elif int(b) == 1:
    c='versicolor'
else:
    c='virginica'
print('预测该花最可能的类型是：',c)

#可视化
tree.plot_tree(clf) 
# 满足 petal width<=0.8的花最有可能是setosa










