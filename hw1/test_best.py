import numpy as np
import pandas as pd
import csv
import sys
# Extract Feachers
testdata = pd.read_csv(sys.argv[1], header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()


test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)

# preprocess wrong data
for n in range(240):
    for feature in range(18):
        for time in range(9):
            if test_x[n][feature * 9 + time] < 0:
                #print(test_x[n][feature * 9 + time])
                neighbor_n = 0
                neighbor_v = 0.0
                for i in range (time - 1, time + 2):
                    if i >=0 and i < 9 and test_x[n][feature * 9 + i] > 0:
                        neighbor_n += 1
                        neighbor_v += test_x[n][feature * 9 + i]
                if neighbor_n == 0:
                    j = time + 1
                    while j < 9:
                        if test_x[n][feature * 9 + j] > 0:
                            neighbor_n = 1
                            neighbor_v = test_x[n][feature * 9 + j]
                            break
                        else:
                            j += 1
                test_x[n][feature * 9 + time] = neighbor_v/neighbor_n
                #print(test_x[n][feature * 9 + time])


"""mean_x = np.mean(test_x, axis = 0) #18 * 9 
std_x = np.std(test_x, axis = 0) #18 * 9 
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]"""
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

# predict
w = np.load('weight_best.npy')
ans_y = np.dot(test_x, w)
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