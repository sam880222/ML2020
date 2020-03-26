import numpy as np
import sys
# Some parameters for training    
max_iter = 70
batch_size = 16
learning_rate = 0.06
c = 0.00025#0.0263
np.random.seed(0)
#raw_data_path = sys.argv[1]
X_train_fpath = sys.argv[1]
Y_train_fpath = sys.argv[2]
validation_portion = 0.2
feature_not01 = [0, 126, 210, 211, 212, 507, 358]
"""feature_to_remove = ['detailed household and family stat', 'country of birth father', 'country of birth mother']#['family members under 18', 'marital stat', 'major industry code', 'major occupation code', 'hispanic origin', 'region of previous residence']#[, 'country of birth father', 'country of birth mother', 'sex', 'live in this house 1 year ago']

with open(raw_data_path) as f:
    x = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = str)    
x = x.T

single_colume = ['age', 'wage per hour', 'capital gains', 'capital losses', 'dividends from stocks', 'weeks worked in year', 'num persons worked for employer']
data = []
for i in range(len(x)):
    if x[i, 0] in single_colume:
        data.append([x[i, 0], 1])
    elif x[i, 0] == 'detailed household and family stat':
        data.append([x[i, 0], len(list(set(x[i, 1:]))) + 2])
    else:
        data.append([x[i, 0], len(list(set(x[i, 1:])))])
data = data[0:-1]
index = dict()
place = 0
for d in data:
    index[d[0]] = [j for j in range(place, place + d[1])]
    place += d[1]
dim = place"""
#print(dim)
#print(index)

def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

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
    
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

def _cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy

def _gradient(X, Y_label, w, c):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1) + w * 2 * c
    #print("grad", pred_error.shape, X.shape)
    return w_grad



# Parse csv files to numpy array
with open(X_train_fpath) as f:
    #next(f)
    label = np.array([f.readline().strip('\n').split(',')[1:]], dtype = str)
    x = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    y = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)


for i in feature_not01[:-1]:
    x = np.concatenate((x, (x[:, i]**2).reshape(-1, 1)), axis = 1).astype(float) 
    x = np.concatenate((x, (x[:, i]**3).reshape(-1, 1)), axis = 1).astype(float)
x = np.concatenate((x, (x[:, 0]**4).reshape(-1, 1)), axis = 1).astype(float)
removed_dims = [286, 299, 302, 10, 198, 370, 16, 78, 309, 374, 102, 95, 166, 455] #[7, 10, 14, 16, 43, 48, 51, 55, 77, 78, 95, 100, 102, 112, 142, 148, 150, 154, 166, 169, 172, 197, 198, 199, 219, 223, 226, 231, 240, 248, 250, 251, 252, 264, 266, 269, 286, 292, 298, 299, 300, 302, 304, 315, 323, 324, 332, 334, 338, 339, 343, 347, 349, 360, 363, 370, 372, 374, 380, 387, 395, 397, 400, 407, 418, 426, 447, 464, 476, 482, 495, 501, 506]#[7, 10, 14, 16, 17, 25, 34, 43, 48, 55, 59, 60, 65, 70, 77, 78, 95, 102, 108, 109, 110, 142, 149, 150, 154, 160, 166, 172, 174, 183, 196, 197, 198, 199, 208, 221, 229, 231, 239, 241, 244, 245, 252, 255, 266, 274, 286, 291, 299, 302, 309, 317, 324, 326, 332, 338, 345, 347, 353, 364, 370, 374, 376, 387, 395, 400, 431, 439, 440, 448, 455, 456, 478]
#removed_dims += list(np.argwhere(label == ' ?')[:, 1]) + list(np.argwhere(label == ' Not in universe')[:, 1])
"""for feature in feature_to_remove:
    removed_dims += index[feature]"""
removed_dims = list(set(removed_dims))
#print(removed_dims)
x = np.delete(x, removed_dims, axis = 1)
#x = np.delete(x, [374, 191, 41, 145], axis = 1)
#x = np.delete(x, [329, 308], axis = 1)
# normalize
 
x, y = _shuffle(x, y)
mean_x = np.mean(x[int(len(x)*validation_portion):], axis = 0)  
std_x = np.std(x[int(len(x)*validation_portion):], axis = 0)
std_x[std_x == 0] = 10e-9
for i in range(len(x)):
    x[i] = (x[i] - mean_x) / std_x
x = np.concatenate((x, np.ones([len(x), 1])), axis = 1).astype(float)    #append bias

X_train = x[int(len(x)*validation_portion):]
Y_train = y[int(len(x)*validation_portion):]
X_valid = x[:int(len(x)*validation_portion)]
Y_valid = y[:int(len(x)*validation_portion)]
train_size = X_train.shape[0]
data_dim = X_train.shape[1]
valid_size = X_valid.shape[0]
print(data_dim)
# Zero initialization for weights ans bias
w = np.zeros((data_dim,)) 



"""# Keep the loss and accuracy at every iteration for plotting
train_loss = []
valid_loss = []
train_acc = []
valid_acc = []"""

# Calcuate the number of parameter updates
step = 1

# Iterative training
for epoch in range(max_iter):
    # Random shuffle at the begging of each epoch
    X_train, Y_train = _shuffle(X_train, Y_train)
        
    # Mini-batch training
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx*batch_size:(idx+1)*batch_size]
        Y = Y_train[idx*batch_size:(idx+1)*batch_size]

        # Compute the gradient
        w_grad = _gradient(X, Y, w, c)
            
        # gradient descent update
        # learning rate decay with time
        w = w - learning_rate/np.sqrt(step) * w_grad

        step = step + 1
            
# Compute loss and accuracy of training set and development set
y_train_pred = _f(X_train, w)
Y_train_pred = np.round(y_train_pred)
train_acc = _accuracy(Y_train_pred, Y_train)
#train_loss = _cross_entropy_loss(y_train_pred, Y_train) / train_size

y_valid_pred = _f(X_valid, w)
Y_valid_pred = np.round(y_valid_pred)
valid_acc = _accuracy(Y_valid_pred, Y_valid)
#valid_loss = _cross_entropy_loss(y_valid_pred, Y_valid) / valid_size

#print('Training loss: {}'.format(train_loss))
#print('Validation loss: {}'.format(valid_loss))
print('Training accuracy: {}'.format(train_acc))
print('Validation accuracy: {}'.format(valid_acc))

np.save('mean_best.npy', mean_x)
np.save('std_best.npy', std_x)
np.save('weight_best.npy', w)
# Print out the most significant weights
"""if len(removed_dims) == 0:
    ind = np.argsort(np.abs(w))[::-1]    
    #ind = np.flip(ind)
    thres = np.abs(w[ind[437]]) 
    #thres -= thres % 0.01

    #print(thres)
    r = np.argwhere(np.abs(w) < thres)
    for i in r:
        print(i[0], end = ', ')
else:
    ind = np.argsort(np.abs(w))[::-1]    
    ind = np.flip(ind)
    for i in ind[:10]:
        print(i, w[i])"""
"""ind = np.argsort(np.abs(w))[::-1]    
ind = np.flip(ind)
for i in ind[:50]:
    print(i, w[i])"""