import numpy as np
import sys
np.random.seed(0)
validation_portion = 0.2
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

def _gradient(X, Y_label, w):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    #print("grad", pred_error.shape, X.shape)
    return w_grad


X_train_fpath = sys.argv[1]
Y_train_fpath = sys.argv[2]

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    x = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    y = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)

#print("train size =", train_size)

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
# Zero initialization for weights ans bias
w = np.zeros((data_dim,)) 

# Some parameters for training    
max_iter = 50
batch_size = 12
learning_rate = 0.2

# Keep the loss and accuracy at every iteration for plotting
train_loss = []
valid_loss = []
train_acc = []
valid_acc = []

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
        w_grad = _gradient(X, Y, w)
            
        # gradient descent update
        # learning rate decay with time
        w = w - learning_rate/np.sqrt(step) * w_grad

        step = step + 1
            
    # Compute loss and accuracy of training set and development set
    y_train_pred = _f(X_train, w)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

    y_valid_pred = _f(X_valid, w)
    Y_valid_pred = np.round(y_valid_pred)
    valid_acc.append(_accuracy(Y_valid_pred, Y_valid))
    valid_loss.append(_cross_entropy_loss(y_valid_pred, Y_valid) / valid_size)

print('Training loss: {}'.format(train_loss[-1]))
print('Validation loss: {}'.format(valid_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Validation loss: {}'.format(valid_acc[-1]))

np.save('mean_logistic.npy', mean_x)
np.save('std_logistic.npy', std_x)
np.save('weight_logistic.npy', w)