import csv
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import linear_model

# 将数据读取为数组
def data_reader(data):
    data_arr = []
    for line in data_raw:
        data_arr.append(line)
    # 选取特征
    data_feature = []
    data_y = []
    for i in range(1, len(data_arr)): #将数据的第一行即特征名去掉
        data_y.append(float(data_arr[i][1]))
        # data_arr从第1行开始读取，data_feature从第零行开始append
        # 将第一个元素转换为list才能append
        data_feature.append([float(data_arr[i][2])])
        # 将男性设置为1，女性设置为0
        if data_arr[i][4] == 'male':
            data_feature[i-1].append(1.0)
        else:
            data_feature[i-1].append(0.0)
        # 设置缺失值
        if data_arr[i][5] == '':
            data_feature[i-1].append(np.nan)
        else:
            data_feature[i-1].append(float(data_arr[i][5]))
        data_feature[i-1].append(float(data_arr[i][6]))
        data_feature[i-1].append(float(data_arr[i][7]))
        data_feature[i-1].append(float(data_arr[i][9]))
    return data_feature, data_y

data_raw = csv.reader(open('C:/Song-Code/Practice/train.csv'))
test_raw = csv.reader(open('C:/Song-Code/Practice/test.csv'))
train_data_has_missing_values, train_y = data_reader(data_raw)
test_data_has_missing_values, test_y = data_reader(data_raw)
# 使用均值填补缺失值
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
train_data = imp.fit(train_data_has_missing_values).transform(train_data_has_missing_values)
# test_data = imp.fit(test_data_has_missing_values).transform(test_data_has_missing_values)
# 标准化
train_scaled = StandardScaler().fit(train_data).transform(train_data)
# test_scaled = StandardScaler().fit(test_data).transform(test_data)

support_vm = svm.SVC()
li = linear_model.LogisticRegression()

svm_scores = cross_val_score(support_vm, train_scaled, train_y, cv = 10)
li_scores = cross_val_score(li, train_scaled, train_y, cv = 10)
print(svm_scores.mean())
print(li_scores.mean())
