import os
import numpy
import matplotlib.pyplot as mtplot
from urllib.request import urlretrieve

# load data
def load_mnist():
    d = "mnist.npz"
    if not os.path.exists(d):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        print("downloading data from", url, "...")
        urlretrieve(url, d)
    f = numpy.load(d)
    x_train, y_train = f['x_train'], f['y_train']
    x_test,  y_test  = f['x_test'],  f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

print("loading data...")
(x_train_, y_train_), (x_test_, y_test_) = load_mnist()

# preprocess: keep only classes 4 and 9, flatten and normalise
def preprocess_data(x_data, y_data):
    filt   = numpy.isin(y_data, [4, 9]) # keep only digits 4 and 9
    x_f    = x_data[filt]
    y_f    = y_data[filt]
    x_flat = x_f.reshape(x_f.shape[0], -1) # flatten 28x28 -> 784
    x_norm = x_flat.astype(numpy.float64) / 255.0  # normalise to [0,1]
    y_bin  = numpy.where(y_f == 4, -1.0, 1.0) # 4 -> -1, 9 -> +1 (float for regression)
    return x_norm, y_bin

X_train, y_train = preprocess_data(x_train_, y_train_) # (N_train, 784)
X_test,  y_test  = preprocess_data(x_test_,  y_test_) # (N_test,  784)
print("train:", X_train.shape, ", test:", X_test.shape)

# train/val split: 1000 per class aside for validation
idx_neg = numpy.where(y_train == -1)[0] # indices for digit 4
idx_pos = numpy.where(y_train ==  1)[0] # indices for digit 9

val_idx = numpy.concatenate([idx_neg[:1000], idx_pos[:1000]])  # 1000 from each class
tr_idx  = numpy.concatenate([idx_neg[1000:], idx_pos[1000:]])  # rest for training

X_val, y_val = X_train[val_idx], y_train[val_idx] # validation set
X_tr,  y_tr  = X_train[tr_idx],  y_train[tr_idx]  # training set
print("train after split:", X_tr.shape, ", val:", X_val.shape)

# pca: fit only on training set
def pca(X, n): # X is (N, 784)
    mew = numpy.mean(X, axis=0, keepdims=True)  # mean of every feature (1, 784)
    Xc  = X - mew # centre the data
    N   = X.shape[0]
    S   = (Xc.T @ Xc) / (N - 1) # cov matrix (784, 784)
    eigval, eigvec = numpy.linalg.eigh(S) # eigh for symmetric matrix
    eigval = numpy.real(eigval)
    eigvec = numpy.real(eigvec)
    i      = numpy.argsort(eigval)[::-1] # sort descending
    eigvec = eigvec[:, i]
    Up     = eigvec[:, :n] # top n eigenvectors (784, n)
    Y      = Xc @ Up # project data (N, n)
    return Y, Up, mew

print("running PCA to 5 dims...")
X_tr_pca,  Up, mew_tr = pca(X_tr, n=5) # fit pca on training only
X_val_pca  = (X_val  - mew_tr) @ Up # project val  with same pca
X_test_pca = (X_test - mew_tr) @ Up # project test with same pca

# regression stump: find the best split minimising SSR
def ssr(y_left, y_right):
    sl = numpy.sum((y_left  - y_left.mean())  ** 2) if len(y_left)  > 0 else 0 # ssr in left leaf
    sr = numpy.sum((y_right - y_right.mean()) ** 2) if len(y_right) > 0 else 0 # ssr in right leaf
    return sl + sr

def train_stump_ssr(X, r): # r = current residuals (or abs-loss labels), shape (n,)
    n, d = X.shape
    best_ssr = numpy.inf
    best_info = {}
    for feat in range(d): # try every pca dimension
        vals = X[:, feat]
        sorted_unique = numpy.unique(vals)
        midpoints = (sorted_unique[:-1] + sorted_unique[1:]) / 2  # midpoints between consecutive values
        if len(midpoints) > 1000: # randomly pick 1000 midpoints to speed up
            midpoints = numpy.random.choice(midpoints, 1000, replace=False)
        for theta in midpoints:
            left_mask = vals <= theta
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0: # skip degenerate splits
                continue
            s = ssr(r[left_mask], r[right_mask]) # compute ssr for this split
            if s < best_ssr:
                best_ssr  = s
                best_info = {'feat': feat, 'theta': theta, 'left_val':  r[left_mask].mean(),  'right_val': r[right_mask].mean()}# leaf prediction = mean of residuals
    return best_info

def predict_stump_reg(X, stump):
    vals = X[:, stump['feat']] # extract the feature this stump splits on
    return numpy.where(vals <= stump['theta'], stump['left_val'], stump['right_val']) # shape (n,)

def mse(x, y):
    return numpy.mean((x - y) ** 2) # mean squared error

def run_gb(eta, T=300):
    F_tr   = numpy.zeros(len(y_tr)) # cumulative prediction on train, start at 0
    F_val  = numpy.zeros(len(y_val))# cumulative prediction on val
    F_test = numpy.zeros(len(y_test))# cumulative prediction on test
    tr_mses, val_mses, test_mses = [], [], []

    for t in range(T):
        residuals  = y_tr - F_tr    # r_i = y_i - F(x_i), negative gradient of l2
        abs_labels = numpy.sign(residuals)# sign(r_i) in {-1,+1}, as done for absolute loss

        stump  = train_stump_ssr(X_tr_pca, abs_labels) # fit stump to minimise SSR on abs labels
        h_tr   = predict_stump_reg(X_tr_pca,   stump)
        h_val  = predict_stump_reg(X_val_pca,  stump)
        h_test = predict_stump_reg(X_test_pca, stump)

        F_tr   += eta * h_tr 
        F_val  += eta * h_val 
        F_test += eta * h_test 

        tr_mses.append(mse(y_tr, F_tr))    # mse = mean((y - F)^2)
        val_mses.append(mse(y_val, F_val))
        test_mses.append(mse(y_test, F_test))

    return tr_mses, val_mses, test_mses

# main run with eta=0.01
print("running gradient boosting with eta=0.01 ...")
tr_mses, val_mses, test_mses = run_gb(eta=0.01, T=300)

best_t = int(numpy.argmin(val_mses)) # iteration with lowest val mse
print("best iteration:", best_t + 1)
print(" train mse :", round(tr_mses[best_t], 6))
print(" val   mse :", round(val_mses[best_t], 6))
print(" test  mse :", round(test_mses[best_t], 6))

# plot val mse vs number of stumps
mtplot.figure(figsize=(9, 5))
mtplot.plot(range(1, 301), val_mses, color='steelblue', linewidth=1.5, label='val mse')
mtplot.axvline(best_t + 1, color='red', linestyle='--',
               label='best iter=' + str(best_t + 1) + ' (mse=' + str(round(val_mses[best_t], 4)) + ')')
mtplot.xlabel("number of stumps (T)")
mtplot.ylabel("validation MSE")
mtplot.title("gradient boosting (eta=0.01): val MSE vs number of stumps")
mtplot.legend()
mtplot.tight_layout()
mtplot.show()

# repeat for multiple learning rates (not for demo)
etas = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0]

fig, axes = mtplot.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

print("\nlearning rate comparison (not for demo):")
print("eta    | best_iter | train_mse  | val_mse    | test_mse")
print("-" * 58)

for i, eta in enumerate(etas):
    tr_m, v_m, te_m = run_gb(eta=eta, T=300)
    best = int(numpy.argmin(v_m))
    print(round(eta, 3), "   |", best+1, "        |", round(tr_m[best], 6),
          "  |", round(v_m[best], 6), "  |", round(te_m[best], 6))

    ax = axes[i]
    ax.plot(range(1, 301), tr_m, label='train', color='royalblue',  linewidth=1.2) # train mse curve
    ax.plot(range(1, 301), v_m,  label='val',   color='darkorange', linewidth=1.2) # val mse curve
    ax.plot(range(1, 301), te_m, label='test',  color='green',      linewidth=1.2) # test mse curve
    ax.set_title("eta = " + str(eta))
    ax.set_xlabel("number of stumps")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=7)

mtplot.suptitle("gradient boosting MSE vs stumps for various eta", fontsize=13)
mtplot.tight_layout()
mtplot.show()
