import numpy as np
import csv

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def main_process():
	train_data = csv.reader(open('D:/Kaggle/Titanic/train.csv'))
	train_x, train_y = data_process(train_data)
	train_paras = paras_cal(np.array(train_x), train_y)
	# print(train_paras)
	test_data = csv.reader(open('D:/Kaggle/Titanic/test.csv'))
	tx = []
	for line in test_data:
		tx.append(line[1:2] + line[3:7])
	tx = tx[1:len(tx)]
	for set in tx:
		for i in range(len(set)):
			if set[i] == 'male':
				set[i] = '1'
			elif set[i] == 'female':
				set[i] = '0'
			elif set[i] =='':    #处理缺失数据，重置为0
				set[i] = '0'
			set[i] = float(set[i])
	prob = []
	for xi in tx:
		prob.append(sigmoid(sum(train_paras * xi)))
	print(prob)

def data_process(file):
	tdx = []
	tdy = []
	for line in file:
		tdx.append(line[2:3] + line[4:8])
		tdy.append(line[1])
	tdx = tdx[1:len(tdx)]
	tdy = tdy[1:len(tdy)]
	for set in tdx:
		for i in range(len(set)):
			if set[i] == 'male':
				set[i] = '1'
			elif set[i] == 'female':
				set[i] = '0'
			elif set[i] =='':    #处理缺失数据，重置为0
				set[i] = '0'
			set[i] = float(set[i])
	for j in range(len(tdy)):
		tdy[j] = float(tdy[j])
	return tdx, tdy

def paras_cal(x_matrix, y_matrix, iter_num=150):
	m, n = np.shape(x_matrix)
	paras = np.ones(n)
	for j in range(iter_num):
		data_index = list(range(m))
		for i in range(m):
			alpha = 4 / (1.0 + j + i) + 0.0001
			rand_index = int(np.random.uniform(0 , len(data_index)))
			h = sigmoid(sum(x_matrix[rand_index] * paras))
			error = y_matrix[rand_index] - h
			paras = paras + error * alpha * x_matrix[rand_index]
			del(data_index[rand_index])
	return paras


def logit_prob(x, paras):
	prob = sigmoid(sum(x * paras))
	if prob > 0.5:
		return 1
	else:
		return 0

main_process()
