import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier

def return_na_index(series):
    count = 0
    index = []
    for num in series.isna():
        if (num): index.append(count)
        count += 1
    return index

raw_train = pd.read_csv("C:/Song-Code/Practice/Titanic/train.csv")
raw_test = pd.read_csv("C:/Song-Code/Practice/Titanic/test.csv")
# print(raw_train.info()) 显示数据信息
# print(raw_test.info()) 显示数据信息

# 将年龄的缺失值按照平均值填充
raw_train['Age'] = raw_train['Age'].fillna(raw_train['Age'].mean())
raw_test['Age'] = raw_test['Age'].fillna(raw_test['Age'].mean())
# test中缺失一个Fare数据
raw_test['Fare'] = raw_test['Fare'].fillna(raw_test['Fare'].mean())
# 将训练数据中Embarked一列中的缺失的两个数据删除
raw_train = raw_train.drop(return_na_index(raw_train['Embarked']), axis=0)
# 删除name，id，survived,ticket列
train = raw_train.drop(['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
test = raw_test.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

# 对Fare进行标准化
def scale(train_series, test_series):
    train_trans = pd.DataFrame(train_series)
    test_trans = pd.DataFrame(test_series)
    scaler = StandardScaler().fit(train_trans)
    train_series = scaler.transform(train_trans)
    test_series = scaler.transform(test_trans)
    return train_series, test_series
train['Fare'], test['Fare'] = scale(train['Fare'], test['Fare'])

# 按年龄将乘客分为child和adult
group_names = ['child', 'adult']
bins = [0, 12, 100]
age_bin_train = pd.cut(train['Age'], bins=bins, labels=group_names)
age_bin_test = pd.cut(test['Age'], bins=bins, labels=group_names)
train['Age'] = age_bin_train
test['Age'] = age_bin_test

# cabin设置成虚拟变量数量太多，所以分为有cabin数据(1)和无cabin数据(0)
def cabin_change(series):
    cabin = []
    for cab in series.isna():
        if cab: cabin.append(0)
        else: cabin.append(1)
    return cabin
train['Cabin'] = cabin_change(train['Cabin'])
test['Cabin'] = cabin_change(test['Cabin'])

# 设置虚拟变量
train = pd.get_dummies(train, columns=['Age', 'Pclass', 'SibSp', 'Parch', 'Cabin', 'Sex', 'Embarked'])
test = pd.get_dummies(test, columns=['Age', 'Pclass', 'SibSp', 'Parch', 'Cabin', 'Sex', 'Embarked'])

# test的parch有9，而train的parch中没有9，所以在test去掉parch_9
test = test.drop(['Parch_9'], axis=1)

# 支持向量机
# scores = []
# c_array = np.linspace(0.1, 1, 10)
# for c in c_array:
#     svc = SVC(C=c)
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# for kernel in kernels:
#     svc = SVC(kernel=kernel)
#     score = cross_validate(svc, train, raw_train['Survived'], cv=10)
#     scores.append(score['test_score'].mean())
# plt.scatter(c_array, scores)
# plt.scatter(kernels, scores)
# plt.show()

# 通过上面的C选择，C=1, kernel=linear,score最高
# 通过learning曲线看模型是否过拟合
svc_selected = SVC(kernel='linear')
train_size, train_score, test_scores = learning_curve(svc_selected, train,
                                        raw_train['Survived'], train_sizes=np.linspace(0.1, 1, 10), cv=5)
plt.plot(train_size, train_score.mean(axis=1), 'r--', train_size, test_scores.mean(axis=1), 'g--')
plt.show()


svc_bagging = BaggingClassifier(svc_selected)
svcb_model = svc_bagging.fit(train, raw_train['Survived'])
preditions = svcb_model.predict(test)
result = pd.DataFrame({'PassengerId': raw_test['PassengerId'], 'Survived':preditions})
result.to_csv('C:/Song-Code/Practice/Titanic/Titanic_Solution.csv', index=False)
