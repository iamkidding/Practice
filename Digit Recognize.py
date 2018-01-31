import numpy as np
import pandas as pd
from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection import cross_val_score

raw_train = pd.read_csv('C:/Song-Code/Practice/Digit Recognizer/train.csv')
raw_test = pd.read_csv('C:/Song-Code/Practice/Digit Recognizer/test.csv')
labels = raw_train['label'][:5000]
data_train = raw_train.iloc[:5000, 1:]

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
cv_max = 0
ker = ''
for kernel in kernels:
    svc = SVC(kernel=kernel)
    cv_score = cross_val_score(svc, data_train, labels, cv=10)
    if (cv_max < cv_score.mean()):
        cv_max = cv_score.mean()
        ker = kernel
print('%s的核得到的cv分数最高，为%d' % (ker, cv_max))
# 通过cv，核为poly的cv分数最高
svc = SVC(kernel='poly')
cv_score = cross_val_score(svc, data_train, labels, cv=10)
print(cv_score.mean())
svc_model = svc.fit(data_train, labels)
predictions = svc_model.predict(raw_test)
sub = pd.DataFrame({'ImageId': pd.Series(range(1, 28001)), 'Label':predictions})
sub.to_csv('C:/Song-Code/Practice/Digit Recognizer/submission.csv', index=False)

# 使用knn需要的时间太长，
# 降低KNN的搜索复杂度，有3种通用的方法：1、降维法；2、预建结构法；3、训练集裁剪法。以后试试
# 根据cv选择k值
# cv_max = 0
# k = 0
# for i in range(3, 20):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     cv = cross_val_score(knn, data_train, raw_train['label'], cv=5)
#     print(i)
#     if (cv > cv_max):
#         cv_max = cv
#         k = i
# print(cv_max, k)
