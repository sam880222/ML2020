import sys
import numpy as np
def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w))

def _predict(X, w):
    # This function returns a truth value prediction for each row of X 
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w)).astype(np.int)

X_test_fpath = sys.argv[1]
output_fpath = sys.argv[2]

with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

mean_x = np.load('mean_logistic.npy')
std_x = np.load('std_logistic.npy')
w = np.load('weight_logistic.npy')

# normalize
for i in range(X_test.shape[0]):
    X_test[i] = (X_test[i] - mean_x) / std_x
X_test = np.concatenate((X_test, np.ones([X_test.shape[0], 1])), axis = 1).astype(float)    #append bias

# Predict testing labels
predictions = _predict(X_test, w)
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))


# Print out the most significant weights
ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])