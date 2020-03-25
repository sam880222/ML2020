import numpy as np
import sys

np.random.seed(0)
X_train_fpath = sys.argv[1]
Y_train_fpath = sys.argv[2]

def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])
   
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    x = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    y = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
y[y == 0] = -1
validation_portion = 0.2
x, y = _shuffle(x, y)
x_train = x[int(len(x)*validation_portion):]
y_train = y[int(len(x)*validation_portion):]
x_valid = x[:int(len(x)*validation_portion)]
y_valid = y[:int(len(x)*validation_portion)]
train_size = x_train.shape[0]
data_dim = x_train.shape[1]
#print("train size =", train_size)


# Some parameters for training    
max_iter = 100
u = np.ones(train_size)/train_size
x_sorted = np.sort(x_train, axis = 0)
thres = []
for i in x_sorted:
    i = np.unique(i)
    thres.append([-10e9] + [0.5*(i[j+1] - i[j]) for j in range(len(i)-1)])
# Keep the loss and accuracy at every iteration for plotting
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

alpha = []
i_t = []
s_t = []
theta_t = []
U = []

E_train =  np.zeros(max_iter)
E_val =  np.zeros(max_iter)
G_train = np.zeros(train_size)
G_val = np.zeros(len(x_valid))
# Iterative training
for t in range(max_iter):
    print("iter #" + str(t) + ':')
    i_opt = 0
    s_opt = -1.0
    theta_opt = -10.0e9
    err_min = 10.0e9
    Ein_min = 10.0e9
    for i in range(data_dim):
        for s in [-1.0, +1.0]:
            for theta in thres[i]:              
                x_i = np.sign(np.sign(x_train[:, i] - theta) + 0.5) * s * y_train
                #print(x_i)
                err = np.sum(u[x_i < 0])
                if err < err_min:
                    i_opt = i
                    s_opt = s
                    theta_opt = theta
                    err_min = err
    epsilon = err_min/np.sum(u)         
    #print("iter #" + str(k) + ": " + Ein_min/train_size)
    alpha.append(np.log((1-epsilon)/epsilon)*0.5)
    i_t.append(i_opt)
    s_t.append(s_opt)
    theta_t.append(theta_opt)
    #print(minerr/N_train)
    reweight_para = ((1-epsilon)/epsilon)**0.5
    for a in range(train_size):
        if s_opt*np.sign(np.sign(x_train[a][i_opt]-theta_opt)+0.5)*y_train[a] < 0:
            u[a] *= reweight_para
        else:
            u[a] /= reweight_para
    U.append(np.sum(u))
    G_train += np.sign(np.sign(x_train[:, i_t[t]]-theta_t[t])+0.5)* alpha[t] * s_t[t]
    G_val += np.sign(np.sign(x_valid[:, i_t[t]]-theta_t[t])+0.5) * alpha[t] * s_t[t]
    #print(G_train, y_train)
    tmp = np.sign(np.sign(G_train) + 0.5) * y_train
    E_train[t] += len(tmp[tmp < 0])
    tmp = np.sign(np.sign(G_val) + 0.5) * y_valid
    E_val[t] += len(tmp[tmp < 0])
   
    E_train[t] /= train_size
    E_val[t] /= len(x_valid)
    print("U: " + str(U[t]) + ", E_train: " + str(E_train[t]) + ", E_valid: " + str(E_val[t]))
    print(u)

print('Training accuracy: {}'.format(1-E_train[-1]))
print('Validation accuracy: {}'.format(1-E_val[-1]))
module = np.zeros((4, max_iter))
module[0, :] = alpha
module[1, :] = i_t
module[2, :] = s_t
module[3, :] = theta_t
np.save("AdaBoost.npy", module)