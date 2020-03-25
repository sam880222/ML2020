import numpy as np
import sys
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from tensorflow.keras import backend
from sklearn import preprocessing
from keras import losses
from keras import metrics
# Some parameters for training    
max_iter = 50
batch_size = 12
learning_rate = 0.2
np.random.seed(0)
raw_data_path = sys.argv[1]
X_train_fpath = sys.argv[2]
Y_train_fpath = sys.argv[3]
feature_to_remove = ['detailed household and family stat', 'country of birth father', 'country of birth mother']#['family members under 18', 'marital stat', 'major industry code', 'major occupation code', 'hispanic origin', 'region of previous residence']#[, 'country of birth father', 'country of birth mother', 'sex', 'live in this house 1 year ago']
validation_portion = 0.2
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
dim = place
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

def _gradient(X, Y_label, w):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
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

removed_dims = list(np.argwhere(label == ' ?')[:, 1]) + list(np.argwhere(label == ' Not in universe')[:, 1])
for feature in feature_to_remove:
    removed_dims += index[feature]
removed_dims = list(set(removed_dims))
#print(removed_dims)
#x = np.delete(x, removed_dims, axis = 1)

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



# Keep the loss and accuracy at every iteration for plotting
train_loss = []
valid_loss = []
train_acc = []
valid_acc = []

# Calcuate the number of parameter updates
step = 1

# Iterative training
model = Sequential()
model.add(Dense(28, activation='sigmoid', input_dim = 511))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#model.add(Dense(units = 1, kernel_initializer = 'he_normal'))
model.compile(optimizer=Adam(lr=0.0003), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

model.fit(X_train, Y_train, epochs = 5, batch_size = 8)

Y_train_pred = model.predict_classes(X_train, batch_size = 8).reshape(1, -1)[0]
print(Y_train_pred)
print(Y_train)
train_acc.append(_accuracy(Y_train_pred, Y_train))
#train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

Y_valid_pred = model.predict_classes(X_valid, batch_size = 8).reshape(1, -1)[0]
valid_acc.append(_accuracy(Y_valid_pred, Y_valid))
#valid_loss.append(_cross_entropy_loss(y_valid_pred, Y_valid) / valid_size)

#print('Training loss: {}'.format(train_loss[-1]))
#print('Validation loss: {}'.format(valid_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Validation loss: {}'.format(valid_acc[-1]))
"""

np.save('weight_best.npy', w)
# Print out the most significant weights
ind = np.argsort(np.abs(w))[::-1]
ind = np.flip(ind)
with open(X_train_fpath) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(i, features[i], w[i])"""
np.save('mean_best.npy', mean_x)
np.save('std_best.npy', std_x)
model.save('NNet_model.h5')
