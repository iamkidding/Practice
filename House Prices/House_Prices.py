import pandas as pd
import seaborn as sns
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
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
plt.show()
corr = train_raw.corr()
f, ax = plt.subplots()
sns.heatmap(corr, vmax=0.8, square=True)
plt.show()
# most_corr = corr.nlargest(10, 'SalePrice') #返回的是包括所有变量的，并在SalePrice那一列按与SalePrice相关系数大小排序的前10行，DataFrame
most_corr = corr.nlargest(15, 'SalePrice')['SalePrice'] #只取SalePrice那一列
train_corr = train_raw.loc[:, ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF','TotRmsAbvGrd', 'YearBuilt']]
test_corr = test_raw.loc[:, ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF','TotRmsAbvGrd', 'YearBuilt']]
# print(train_corr.info())

rf = RandomForestRegressor()
svr = SVR()
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
train = train_na.fillna(train_na.mean())
test = test_na.fillna(train_na.mean())

# 设置虚拟变量
train_dum = pd.get_dummies(train)
test_dum = pd.get_dummies(test)

# iter = 0
colnames_selected = []
# while (iter < 10):
rf = RandomForestRegressor()
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
# print(colnames_selected)
# colnames_selected = colnames_selected[:25] #先将虚拟变量的列去除 # 这样不行，每次计算的colnames会发生变化
train_selected = train_dum[colnames_selected]
test_selected = test_dum[colnames_selected]
# print(test_dum.info())
# print(test_selected.info())
clf_selected = rf.fit(train_selected, train_raw['SalePrice']) #使用选择的特征重新计算模型
predictions = clf_selected.predict(test_selected)
# result = pd.DataFrame({'Id': test_raw['Id'], 'SalePrice': predictions})
# result.to_csv('C:/Song-Code/Practice/House Prices/submission.csv', index=False)

# 此例中将train和test分别做虚拟变量变换会导致train和test的变量个数不一致，train270，test254
all_data = pd.concat([train, test]) # 这一步不会修改原DataFrame的index，得用reset_index()手动修改
all_data_dum = pd.get_dummies(all_data)
all_data_dum = all_data_dum.reset_index()
train_all = all_data_dum.loc[:1459,]
test_all = all_data_dum.loc[1460:,]
#标准化
scale = StandardScaler()
scale_para = scale.fit(train_all)
train_sca = scale.fit_transform(train_all, scale_para)
test_sca = scale.fit_transform(test_all, scale_para)

# 测试不同alpha值的CV，选出最合适的
alphas = [20, 30, 50, 75, 100, 110, 125, 130,150]
rd_means = []
ls_means = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ls = Lasso(alpha=alpha)
    rd_cv = cross_val_score(ridge, train_sca, train_raw['SalePrice'], cv=10)
    ls_cv = cross_val_score(ls, train_sca, train_raw['SalePrice'], cv=10)
    rd_means.append(rd_cv.mean())
    ls_means.append(ls_cv.mean())
plt.figure(1)
plt.subplot(211)
plt.plot(alphas, rd_means, 'b')
plt.subplot(212)
plt.plot(alphas, ls_means, 'g')
plt.show()
# print(rd_cv.mean())
# print(rf_cv.mean())
# print(ls_cv.mean())

ri_model = ridge.fit(train_sca, train_raw['SalePrice'])
ri_pre = ri_model.predict(test_sca)
re_sub = pd.DataFrame({'Id':test_raw['Id'], 'SalePrice':ri_pre})
re_sub.to_csv('C:/Song-Code/Practice/House Prices/sub_ridge.csv', index=False)
