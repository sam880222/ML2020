import numpy as np
import pandas as pd
import csv
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.externals import joblib
clf = joblib.load('pow3')
trace_hour = 7
power = 3
removed_features = [1, 10, 14, 15, 16, 17]
features = 18 - len(removed_features)
# Extract Feachers
testdata = pd.read_csv(sys.argv[1], header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()


test_x = np.empty([240, features * trace_hour], dtype = float)
for i in range(240):
    tmp = test_data[18 * i: 18* (i + 1), -trace_hour:]
    test_x[i, :] = np.delete(tmp, removed_features, axis = 0).reshape(1, -1)

# preprocess wrong data
for n in range(240):
    for feature in range(features):
        for time in range(trace_hour):
            if test_x[n][feature * trace_hour + time] < 0:
                #print(test_x[n][feature * 9 + time])
                neighbor_n = 0
                neighbor_v = 0.0
                for i in range (time - 1, time + 2):
                    if i >=0 and i < trace_hour and test_x[n][feature * trace_hour + i] > 0:
                        neighbor_n += 1
                        neighbor_v += test_x[n][feature * trace_hour + i]
                if neighbor_n == 0:
                    j = time + 1
                    while j < trace_hour:
                        if test_x[n][feature * trace_hour + j] > 0:
                            neighbor_n = 1
                            neighbor_v = test_x[n][feature * trace_hour + j]
                            break
                        else:
                            j += 1
                test_x[n][feature * trace_hour + time] = neighbor_v/neighbor_n
                #print(test_x[n][feature * 9 + time])

test_x = np.hstack((test_x**(i+1) for i in range(power)))
mean_x = np.load('mean_pow3.npy')
std_x = np.load('std_pow3.npy')
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)


# predict

#w = np.load('weight_pow3.npy')
#ans_y = np.dot(test_x, w)
ans_y = clf.predict(test_x)
ans_y = np.round(ans_y)
ans_y[ans_y < 0] = 0

# output result
with open(sys.argv[2], mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)