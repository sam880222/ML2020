import numpy as np
import sys
X_train_fpath = sys.argv[1]
Y_train_fpath = sys.argv[2]

def _predict(X, w, b):
    z = np.matmul(X, w) + b
    sigmoid = np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))
    return np.round(sigmoid).astype(np.int)

def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)

train_size = X_train.shape[0]
data_dim = X_train.shape[1]
#print("train size =", train_size)

# normalize
mean_x = np.mean(X_train, axis = 0)  
std_x = np.std(X_train, axis = 0)
std_x[std_x == 0] = 10e-9
for i in range(train_size):
    X_train[i] = (X_train[i] - mean_x) / std_x

# Compute in-class mean
X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])
mean_0 = np.mean(X_train_0, axis = 0)
mean_1 = np.mean(X_train_1, axis = 0)  

# Compute in-class covariance
cov_0 = np.zeros((data_dim, data_dim))
cov_1 = np.zeros((data_dim, data_dim))

for x in X_train_0:
    cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
for x in X_train_1:
    cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

# Shared covariance is taken as a weighted average of individual in-class covariance.
cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])

# Compute inverse of covariance matrix.
# Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
# Via SVD decomposition, one can get matrix inverse efficiently and accurately.
u, s, v = np.linalg.svd(cov, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)

# Directly compute weights and bias
w = np.dot(inv, mean_1 - mean_0)
b =  (-0.5) * np.dot(mean_1, np.dot(inv, mean_1)) + 0.5 * np.dot(mean_0, np.dot(inv, mean_0)) + np.log(float(X_train_1.shape[0]) / X_train_0.shape[0]) 

# Compute accuracy on training set
Y_train_pred = _predict(X_train, w, b)
print('Training accuracy: {}'.format(_accuracy(Y_train_pred, Y_train)))

w = np.concatenate((w, np.array([b])))
np.save('mean_generative.npy', mean_x)
np.save('std_generative.npy', std_x)
np.save('weight_generative.npy', w)