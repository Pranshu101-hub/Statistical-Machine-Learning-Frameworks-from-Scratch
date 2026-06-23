import os
import numpy as np
import matplotlib.pyplot as mtplot

#load
def load_fashion():
    fname = "fashion_mnist.npz"
    if os.path.exists(fname):
        print("loading fashion mnist from cache...")
        d = np.load(fname)
        return (d['x_train'], d['y_train']), (d['x_test'], d['y_test'])
    try:
        import tensorflow.keras as keras
        print("downloading via tensorflow...")
        (x_tr, y_tr), (x_te, y_te) = keras.datasets.fashion_mnist.load_data()
        np.savez(fname, x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te)
        return (x_tr, y_tr), (x_te, y_te)
    except Exception:
        raise FileNotFoundError("please download fashion_mnist.npz manually and place it here")

(x_train_, y_train_), (x_test_, y_test_) = load_fashion()

#preprocess
def preprocess_data(x_data, y_data):
    filt = np.isin(y_data, [0, 1, 2]) # keep only classes 0,1,2
    x_f = x_data[filt]
    y_f = y_data[filt]
    x_flat = x_f.reshape(x_f.shape[0], -1) # flatten 28x28 -> 784
    x_norm = x_flat.astype(np.float64) / 255.0    # normalise to [0,1]
    return x_norm, y_f.astype(float) # labels as float for regression

X_train, y_train = preprocess_data(x_train_, y_train_)
X_test,  y_test  = preprocess_data(x_test_,  y_test_)
print("train:", X_train.shape, ", test:", X_test.shape)

# pca
def pca(X, n_components):
    mew = np.mean(X, axis=0, keepdims=True) # feature-wise mean
    Xc = X - mew # centre the data
    N = X.shape[0]
    S = (Xc.T @ Xc) / (N - 1) # covariance matrix

    eigval, eigvec = np.linalg.eigh(S) # eigendecomposition
    eigval = np.real(eigval)
    eigvec = np.real(eigvec)

    i = np.argsort(eigval)[::-1] # sort biggest first
    eigvec = eigvec[:, i]

    Up = eigvec[:, :n_components] # top n eigenvectors
    Y = Xc @ Up # projected data (N, n_components)
    return Y, Up, mew

print("running PCA to 10 dims...")
X_tr, Up, mew_tr = pca(X_train, n_components=10)
X_te = (X_test - mew_tr) @ Up   # project test set using same PCA

#regression stump helpers  
def ssr(y_left, y_right):
    #sum of squared residuals = variance within each region * N
    sl = np.sum((y_left - y_left.mean())  ** 2) if len(y_left) > 0 else 0
    sr = np.sum((y_right - y_right.mean()) ** 2) if len(y_right) > 0 else 0
    return sl + sr

def find_best_stump(X, y):
    # try every feature and every midpoint threshold, pick the one with lowest SSR
    n, d = X.shape
    best_ssr = np.inf
    best_f, best_t = 0, 0.0
    for f in range(d):
        vals = np.sort(np.unique(X[:, f])) # sorted unique values
        if len(vals) < 2:
            continue
        thresholds = (vals[:-1] + vals[1:]) / 2.0   # midpoints between consecutive values
        for t in thresholds:
            left_mask = X[:, f] <= t
            left  = y[left_mask]
            right = y[~left_mask]
            s     = ssr(left, right) # compute SSR for this split
            if s < best_ssr:
                best_ssr, best_f, best_t = s, f, t   # save best split
    # get mean prediction for each region
    mean_l = y[X[:, best_f] <= best_t].mean()
    mean_r = y[X[:, best_f] >  best_t].mean()
    return best_f, best_t, mean_l, mean_r #the 4 values defining a stump

def predict_stump(stump, X):
    f, t, mean_l, mean_r = stump # if sample goes left -> predict mean_l, else predict mean_r
    return np.where(X[:, f] <= t, mean_l, mean_r)

def mse(x, y):
    return np.mean((x - y) ** 2)   #mean squared error

#single regression stump h1
print("training single stump h1...")
stump1 = find_best_stump(X_tr, y_train)
f1, t1, ml1, mr1 = stump1
print("best feature:", f1, ", threshold:", round(t1, 4))
print("mean left:", round(ml1, 4), ", mean right:", round(mr1, 4))

pred_te1 = predict_stump(stump1, X_te)
mse_te1 = mse(pred_te1, y_test)
print("single stump test MSE:", round(mse_te1, 4))

#bagging: 5 bootstrap stumps
print("\nbagging with 5 stumps...")
N = len(X_tr)
n_trees =5
rng = np.random.RandomState(67)  #fixed seed

boot_stumps = []
oob_mse_list = []

for b in range(n_trees):
    boot_idx = rng.randint(0, N, size=N) # sample with replacement
    oob_idx = np.setdiff1d(np.arange(N), boot_idx)  # samples not in bootstrap
    stump_b = find_best_stump(X_tr[boot_idx], y_train[boot_idx]) #fit stump
    boot_stumps.append(stump_b)

    oob_preds = predict_stump(stump_b, X_tr[oob_idx])   # predict on oob samples
    # correct oob error: use training labels for oob samples
    oob_err = mse(oob_preds, y_train[oob_idx])
    oob_mse_list.append(oob_err)
    print("stump", b+1, ": OOB MSE =", round(oob_err, 4))

print("average OOB MSE across 5 stumps:", round(np.mean(oob_mse_list), 4))

# bagged prediction = average of all 5 stump predictions
bag_te_preds = np.mean(
    np.column_stack([predict_stump(s, X_te) for s in boot_stumps]), axis=1)
mse_bag = mse(bag_te_preds, y_test)
print("bagging test MSE:", round(mse_bag, 4))

#comparison 
sort_idx = np.argsort(y_test)   # sort by true label so plot looks clean
mtplot.figure(figsize=(12, 5))
mtplot.plot(y_test[sort_idx],"k-",  alpha=0.3, linewidth=1,  label="true labels")
mtplot.plot(pred_te1[sort_idx],"b.",  alpha=0.4, markersize=2, label="single stump (MSE=" + str(round(mse_te1, 3)) + ")")
mtplot.plot(bag_te_preds[sort_idx],"r.",  alpha=0.4, markersize=2, label="bagging      (MSE=" + str(round(mse_bag, 3)) + ")")
mtplot.xlabel("test sample (sorted by true label)")
mtplot.ylabel("predicted value")
mtplot.title("single stump vs bagging — test predictions")
mtplot.legend()
mtplot.grid(True, alpha=0.3)
mtplot.tight_layout()
mtplot.show()

#summary
print("\nsingle stump test MSE:", round(mse_te1, 4))
print("bagging test MSE:", round(mse_bag, 4))
print("bagging reduces variance by averaging 5 stumps trained on different bootstrap samples")
print("each stump predicts one of only 2 values; averaging them gives a smoother prediction")