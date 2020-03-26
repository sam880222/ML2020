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
    #next(f)
    label = np.array([f.readline().strip('\n').split(',')[1:]], dtype = str)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

mean_x = np.load('mean_best.npy')
std_x = np.load('std_best.npy')
w = np.load('weight_best.npy')
feature_not01 = [0, 126, 210, 211, 212, 507, 358]
for i in feature_not01[:-1]:
    X_test = np.concatenate((X_test, (X_test[:, i]**2).reshape(-1, 1)), axis = 1).astype(float) 
    X_test = np.concatenate((X_test, (X_test[:, i]**3).reshape(-1, 1)), axis = 1).astype(float)
X_test = np.concatenate((X_test, (X_test[:, 1]**4).reshape(-1, 1)), axis = 1).astype(float)
removed_dims = [286, 299, 302, 10, 198, 370, 16, 78, 309, 374, 102, 95, 166, 455] #[7, 10, 14, 16, 17, 25, 34, 43, 48, 55, 59, 60, 65, 70, 77, 78, 95, 102, 108, 109, 110, 142, 149, 150, 154, 160, 166, 172, 174, 183, 196, 197, 198, 199, 208, 221, 229, 231, 239, 241, 244, 245, 252, 255, 266, 274, 286, 291, 299, 302, 309, 317, 324, 326, 332, 338, 345, 347, 353, 364, 370, 374, 376, 387, 395, 400, 431, 439, 440, 448, 455, 456, 478]
#removed_dims += list(np.argwhere(label == ' ?')[:, 1]) + list(np.argwhere(label == ' Not in universe')[:, 1])
removed_dims = list(set(removed_dims))
#print(removed_dims)
X_test = np.delete(X_test, removed_dims, axis = 1)
#X_test = np.delete(X_test, [53, 215, 242], axis = 1)
#X_test = np.delete(X_test, [374, 191, 41, 145], axis = 1)
"""X_test = np.delete(X_test, [168, 247], axis = 1)
X_test = np.delete(X_test, [333, 212, 323], axis = 1)
X_test = np.delete(X_test, [23, 325, 306, 317], axis = 1)"""
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