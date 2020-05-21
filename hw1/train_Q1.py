import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import training data
data = pd.read_csv('./train.csv', encoding = 'big5')
data = data.iloc[:, 3:]
#print(type(data))
data[data == 'NR'] = 0
raw_data = data.to_numpy()

# extract feachers (12 18*(24)*20 data)
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample

"""
# preprocess wrong data
for month in range(12):
    for feature in range(18):
        for time in range(480):
            if month_data[month][feature][time] < 0:
                neighbor_n = 0
                neighbor_v = 0.0
                for i in range (time - 1, time + 2):
                    if i >=0 and i < 480 and month_data[month][feature][i] > 0:
                        neighbor_n += 1
                        neighbor_v += month_data[month][feature][i]
                if neighbor_n == 0:
                    j = time + 1
                    while j < 480:
                        if month_data[month][feature][j] > 0:
                            neighbor_n = 1
                            neighbor_v = month_data[month][feature][j]
                            break
                        else:
                            j += 1
                month_data[month][feature][time] = neighbor_v/neighbor_n
                #print(month_data[month][feature][time])"""

# 471 (X, y) pairs
x = np.empty([12 * 471, 18 * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value
#print(x)
#print(y)

# normalize
mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

# Adagrad
learning_rate = [0.01, 0.1, 1, 10]
color = ['purple', 'green', 'red', 'blue']
dim = 18 * 9 + 1
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
iter_time = 10000
for i in range(4):
    L = []
    w = np.zeros([dim, 1])
    adagrad = np.zeros([dim, 1])
    eps = 0.0000000001
    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
        
            
        if(t % 10 == 0):
            #print(str(t) + ":" + str(loss))
            L.append(loss)
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1

        adagrad += gradient ** 2
        w = w - learning_rate[i] * gradient / np.sqrt(adagrad + eps)
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)
    L.append(loss)
    plt.plot(range(0, iter_time+1, 10), L, color = color[i], label = "rate = " + str(learning_rate[i]))
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig('Q1.png')
plt.show()