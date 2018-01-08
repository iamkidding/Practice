import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

train_raw = pd.read_csv("C:/Song-Code/Practice/Data Science London/train.csv", header=None)
train_labels_raw = pd.read_csv("C:/Song-Code/Practice/Data Science London/trainLabels.csv",
                              header=None, names=['Labels'])
test_raw = pd.read_csv("C:/Song-Code/Practice/Data Science London/test.csv", header=None)
#标准化
scaler = StandardScaler()
para_scale = scaler.fit(train_raw)
train_scaled = scaler.fit_transform(train_raw, para_scale)
test_scaled = scaler.fit_transform(test_raw, para_scale)
# 代入模型中计算的X.shape应为(m,n)，Y.shape应为(m,)（类似一维数组)
# 如果Y.shape为(m,1)，会出现warning
# 在dataframe中的任一列的shape为(m,)
train_labels = pd.Series(train_labels_raw["Labels"])
# print(train_scaled.shape)
# print(train_labels_raw.shape)

logstic = LogisticRegression()
vector = svm.SVC()
li = logstic.fit(train_scaled, train_labels)
sv = vector.fit(train_scaled, train_labels)

lcv = cross_validate(logstic, train_scaled, train_labels, cv=10)
scv = cross_validate(vector, train_scaled, train_labels, cv=10)
print(lcv['test_score'].mean())
print(scv['test_score'].mean())

predictions = sv.predict(test_scaled)
result = pd.DataFrame({"Id":range(1, 9001), "Solution":predictions.astype(np.int32)})
result.to_csv("C:/Song-Code/Practice/Data Science London/Submission.csv", index=False)
