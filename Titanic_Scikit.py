# from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import linear_model
import pandas as pd

data_raw = pd.read_csv(open('C:/Song-Code/Practice/Titanic/train.csv'))
test_raw = pd.read_csv(open('C:/Song-Code/Practice/Titanic/test.csv'))
# 将男性设置为1，女性设置为0
# data_raw = data_raw.replace(['male', 'female'], [1, 0])
# test_raw = test_raw.replace(['male','female'], [1, 0])
# 将Sex，Pclass设置成虚拟变量
# 不设置成虚拟变量，而是直接将sex改成0,1值，最后的泛化结果较差，在kaggle为95%，
# 使用虚拟变量，得出的svm模型可以到44%
dum_sex = pd.get_dummies(data_raw['Sex'], prefix="Sex")
dum_pclass = pd.get_dummies(data_raw['Pclass'], prefix="Pclass")
data_raw = pd.concat([data_raw, dum_sex, dum_pclass], axis=1)
dum_sex_test = pd.get_dummies(test_raw['Sex'], prefix="Sex")
dum_pclass_test = pd.get_dummies(test_raw['Pclass'], prefix="Pclass")
test_raw = pd.concat([test_raw, dum_sex_test, dum_pclass_test], axis=1)

# # 使用均值填补缺失值
data_raw["Age"] = data_raw["Age"].fillna(data_raw["Age"].mean())
test_raw["Age"] = test_raw["Age"].fillna(test_raw["Age"].mean())
test_raw["Fare"] = test_raw["Fare"].fillna(test_raw["Fare"].mean())
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
# # data_raw["Age"] = imp.fit(data_raw["Age"]).transform(data_raw["Age"])
# # data_raw = imp.fit(data_raw).transform(data_raw)

# 标准化
scaler = StandardScaler()
age_2d_train = pd.DataFrame(data_raw["Age"])
age_2d_test = pd.DataFrame(test_raw["Age"])
fare_2d_train = pd.DataFrame(data_raw["Fare"])
fare_2d_test = pd.DataFrame(test_raw["Fare"])
age_scale_para = scaler.fit(age_2d_train)
fare_scale_para = scaler.fit(fare_2d_train)
data_raw["Age_scaled"] = scaler.fit_transform(age_2d_train, age_scale_para)
data_raw["Fare_scaled"] = scaler.fit_transform(fare_2d_train, fare_scale_para)
test_raw["Age_scaled"] = scaler.fit_transform(age_2d_test, age_scale_para)#, age_scale_para_test)
test_raw["Fare_scaled"] = scaler.fit_transform(fare_2d_test, fare_scale_para)#, fare_scale_para_test)
# age_scale_para = scaler.fit(data_raw["Age"])
# 会出现ValueError: Expected 2D array, got 1D array instead:
# 例如数据格式为[1,  2, 3, 4]就会出错，如果把这行数据转换成[[1], [2], [3], [4]]就不会出错了
# 要对上面导致出错的两行代码做出修改：加上.reshape(-1,1）,主要是工具包版本更新造成的，但reshape已经被抛弃了
# 所以对data_raw["Age"]使用其他方式进行shape转换，StandardScaler.fit()的参数.shape应该为(m,n)，不能是1D的(m,)

# 将年龄分组，分为18岁以下，18-60,60以上三组
age_bins = [0, 12, 100]
age_group = ['Child', 'Adult']
data_raw["Age"] = pd.cut(data_raw["Age"], age_bins, labels=age_group)
# data_raw_age = pd.concat([data_raw, pd.DataFrame(Age_cat)], axis=1) #为什么这一步完成后data_raw_age为空
age_dum_train = pd.get_dummies(data_raw['Age'], prefix='Age')
data_raw = pd.concat([data_raw, age_dum_train], axis=1)
test_raw["Age"] = pd.cut(test_raw["Age"], age_bins, labels=age_group)
age_dum_test = pd.get_dummies(test_raw['Age'], prefix='Age')
test_raw = pd.concat([test_raw, age_dum_test], axis=1)

# 选取要包含在模型中的特征
train = data_raw.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin", "Embarked", "Sex",
                       "Age", "Pclass", "Fare", "Age"], axis = 1)
test = test_raw.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked","Sex",
                       "Age", "Pclass", "Fare", "Age"], axis = 1)
# print(data_raw.info())
# print(data_raw.describe())

#模型设置
support_vm = svm.SVC()
# support_vm = svm.SVC(kernel="linear")
li = linear_model.LogisticRegression()

#模型交叉验证得分
svm_scores = cross_val_score(support_vm, train, data_raw["Survived"], cv = 10)
li_scores = cross_val_score(li, train, data_raw["Survived"], cv = 10)
print(svm_scores.mean())
print(li_scores.mean())

#
clf_svm = support_vm.fit(train, data_raw["Survived"])
clf_li = li.fit(train, data_raw["Survived"])
# print(clf_svm)
# # print(clf_li)
# predi_logi = clf_li.predict(test_data)  #kaggle正确率 0.66985
predi_svm = clf_svm.predict(test)
print(predi_svm)
result = pd.DataFrame({'PassengerId':test_raw['PassengerId'].as_matrix(), 'Survived':predi_svm.astype(np.int32)})
result.to_csv("C:/Song-Code/Practice/Titanic/Titanic_logistic_regression_predictions.csv", index=False)
