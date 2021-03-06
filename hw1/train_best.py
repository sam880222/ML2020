import sys
import pandas as pd
import numpy as np

### parameters to be change
np.random.seed(0)
trace_hour = 8
removed_features = [1, 10, 14, 15, 16, 17]#[13, 14, 15, 16, 17]
features = 18 - len(removed_features)

def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

# import training data
data = pd.read_csv('./train.csv', encoding = 'big5')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

# extract feachers (12 18*(24)*20 data)
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample

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
                #print(month_data[month][feature][time])

# 471 (X, y) pairs
x = np.empty([12 * (480 - trace_hour), features * trace_hour], dtype = float)
y = np.empty([12 * (480 - trace_hour), 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > (23 - trace_hour):
                continue
            tmp = month_data[month][:,day * 24 + hour : day * 24 + hour + trace_hour]
            x[month * (480 - trace_hour) + day * 24 + hour, :] = np.delete(tmp, removed_features, axis = 0).reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * (480 - trace_hour) + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + trace_hour] #value
#print(x)
#print(y)


x, y = _shuffle(x, y)
# normalize
validation_portion = 0.2
mean_x = np.mean(x[int(len(x)*validation_portion):], axis = 0) #18 * 9 
std_x = np.std(x[int(len(x)*validation_portion):], axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
x = np.concatenate((np.ones([12 * (480 - trace_hour), 1]), x), axis = 1).astype(float)

#for i in range(x.shape[0]):
#    x = np.concatenate((np.power(x, 2), x), axis = 1).astype(float)

x_train = x[int(len(x)*validation_portion):]
y_train = y[int(len(x)*validation_portion):]
x_valid = x[:int(len(x)*validation_portion)]
y_valid = y[:int(len(x)*validation_portion)]

# Adagrad
dim = features * trace_hour + 1
w = np.zeros([dim, 1])
learning_rate = 10
iter_time = 10000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.mean(np.power(np.dot(x_train, w) - y_train, 2)))#rmse
    #if(t%100==0):
        #print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x_train.transpose(), np.dot(x_train, w) - y_train) #dim*1
  
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)

### save module
np.save('weight_best.npy', w)
np.save('mean_best.npy', mean_x)
np.save('std_best.npy', std_x)

### print losses of training and validation
y_pred = np.dot(x_train, w)
y_pred = np.round(y_pred)
y_pred[y_pred < 0] = 0
loss_train = np.sqrt(np.mean(np.power(y_pred - y_train, 2)))
print("train loss:", loss_train)
if validation_portion > 0:
    y_pred = np.dot(x_valid, w)
    y_pred = np.round(y_pred)
    y_pred[y_pred < 0] = 0
    loss_valid = np.sqrt(np.mean(np.power(y_pred - y_valid, 2)))
    print("validation loss:", loss_valid)

#print weight
data = pd.read_csv('./train.csv', encoding = 'big5')
data = data.iloc[0:18, 2].to_numpy().reshape(18, 1)
data = np.concatenate((np.array([i for i in range(18)]).reshape(18, 1), data), axis = 1).reshape(18, 2)
data = np.delete(data, removed_features, axis = 0)
w = w[1:].reshape(features, trace_hour)
w_mean = np.mean(np.abs(w), axis = 1)
ind = np.argsort(np.abs(w_mean))[::-1]
for i in ind:
    print(data[i], w_mean[i])
#for i in range(features):
    #print(data[i], np.mean(w[i * trace_hour : (i + 1) * trace_hour].reshape(1, -1), axis = 1))
#print(w)

