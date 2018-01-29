import pandas as pd
import seaborn as sns
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

train_raw = pd.read_csv("C:/Song-Code/Practice/House Prices/train.csv")
test_raw = pd.read_csv("C:/Song-Code/Practice/House Prices/test.csv")
# print(train_raw.info())
# print(train_raw.describe())

fig1, ax1 = plt.subplots(2, 2, figsize=(5, 5))
ax1[0, 0].scatter(x=train_raw['LotArea'], y=train_raw['SalePrice'])
ax1[0, 1].scatter(x=train_raw['MSSubClass'], y=train_raw['SalePrice'])
ax1[1, 0].scatter(x=train_raw['OverallQual'], y=train_raw['SalePrice'])
ax1[1, 1].scatter(x=train_raw['OverallCond'], y=train_raw['SalePrice'])
# plt.show()
corr = train_raw.corr()
f, ax = plt.subplots()
sns.heatmap(corr, vmax=0.8, square=True)
# plt.show()
# most_corr = corr.nlargest(10, 'SalePrice') #返回的是包括所有变量的，并在SalePrice那一列按与SalePrice相关系数大小排序的前10行，DataFrame
most_corr = corr.nlargest(15, 'SalePrice')['SalePrice'] #只取SalePrice那一列
train_corr = train_raw.loc[:, ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF','TotRmsAbvGrd', 'YearBuilt']]
test_corr = test_raw.loc[:, ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF','TotRmsAbvGrd', 'YearBuilt']]
# print(train_corr.info())

rf = RandomForestRegressor()
svr = SVR()
ridge = Ridge(alpha=10)
ls = Lasso()
#Cross Validation
rf_cv = cross_val_score(rf, train_corr, train_raw['SalePrice'], cv=10)
svr_cv = cross_val_score(svr, train_corr, train_raw['SalePrice'], cv=10)

model = rf.fit(train_corr, train_raw['SalePrice'])
# print(model.feature_importances_)
test_corr = test_corr.fillna(test_corr.mean())
pre = model.predict(test_corr)
sub = pd.DataFrame({'Id': test_raw['Id'], 'SalePrice': pre})
sub.to_csv('C:/Song-Code/Practice/House Prices/sub.csv', index=False)

# Alley(91)、PoolQC(7)、Fence(281)、MiscFeature(54)、FireplaceQu(770)缺失值太多
train_na = train_raw.drop(["Id", "Alley", "PoolQC", "Fence", "MiscFeature", "FireplaceQu", "SalePrice"], axis=1)
test_na = test_raw.drop(["Id", "Alley", "PoolQC", "Fence", "MiscFeature", "FireplaceQu"], axis=1)
# 填补缺失值，使用前面的值填补(pad),bfill是使用后面的值，mean均值
train = train_na.fillna(method='pad')
test = test_na.fillna(method='pad')

# 设置虚拟变量
train_dum = pd.get_dummies(train)
test_dum = pd.get_dummies(test)

# iter = 0
colnames_selected = []
# while (iter < 10):
rf = RandomForestRegressor(oob_score=True)
clf = rf.fit(train_dum,train_raw['SalePrice']) # 使用全部特征计算模型，然后选择特征
# print(clf.feature_importances_)
i = 0  # 特征的index
feature_selected = 0 # 大于阈值的特征的个数
 # 被选特征的列名统计
for item in clf.feature_importances_:
    if (item > 0.001):
        if (train_dum.columns[i] not in colnames_selected):
            colnames_selected.append(train_dum.columns[i])
        feature_selected += 1
        # print('第%d个特征重要性为%f' % (i, item))
    i += 1
# iter += 1
# print(i);
# print(feature_selected)
print(colnames_selected)
# colnames_selected = colnames_selected[:25] #先将虚拟变量的列去除 # 这样不行，每次计算的colnames会发生变化
train_selected = train_dum[colnames_selected]
test_selected = test_dum[colnames_selected]
# print(test_dum.info())
# print(test_selected.info())
clf_selected = rf.fit(train_selected, train_raw['SalePrice']) #使用选择的特征重新计算模型
predictions = clf_selected.predict(test_selected)
# result = pd.DataFrame({'Id': test_raw['Id'], 'SalePrice': predictions})
# result.to_csv('C:/Song-Code/Practice/House Prices/submission.csv', index=False)

rd_cv = cross_val_score(ridge, train_selected, train_raw['SalePrice'], cv=10)
ls_cv = cross_val_score(ls, train_selected, train_raw['SalePrice'], cv=10)
print(rd_cv.mean())
print(ls_cv.mean())
