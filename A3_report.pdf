import os
import numpy
import matplotlib.pyplot as mtplot
from urllib.request import urlretrieve
from sklearn.linear_model import Lasso

#load data
def load_mnist():
    d = "mnist.numpyz"
    if not os.path.exists(d):
        # Updated URL
        url = "https://s3.amazonaws.com/img-datasets/mnist.npz"
        print(f"Downloading data from {url}...")
        urlretrieve(url, d)
    
    f = numpy.load(d)
    x_train, y_train = f['x_train'], f['y_train']
    x_test,  y_test  = f['x_test'],  f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

print("loading data...")
(x_train_, y_train_), (x_test_, y_test_) = load_mnist()

#preprocess 
def preprocess_data(x_data, y_data):
    filt = numpy.isin(y_data, [0, 1, 2])   # keep only classes 0, 1, 2
    x_f  = x_data[filt]
    y_f  = y_data[filt]
    x_flat = x_f.reshape(x_f.shape[0], -1)          # flatten 28x28 -> 784
    x_norm = x_flat.astype(numpy.float64) / 255.0       # normalise to [0,1]
    return x_norm, y_f

X_train, y_train = preprocess_data(x_train_, y_train_)  # shape (N_train, 784)
X_test,  y_test  = preprocess_data(x_test_,  y_test_)   # shape (N_test,  784)
print(f"train: {X_train.shape}, test: {X_test.shape}")

#pca
def pca(X, n): #X is (N, 784) 
    mew = numpy.mean(X, axis=0, keepdims=True)   # mean of every feature (1, 784)
    Xc = X - mew #centering data
    N = X.shape[0] #number of samples
    S = (Xc.T @ Xc) / (N - 1) #cov matrix (784, 784)
    eigval, eigvec = numpy.linalg.eigh(S) #get eigenvalues and eigenvectors
    eigval = numpy.real(eigval) #keep real parts
    eigvec = numpy.real(eigvec)
    i = numpy.argsort(eigval)[::-1] #sort decending
    eigvec = eigvec[:, i] #sort eigenvectors by eigenvalues
    Up = eigvec[:, :n] #take n eigenvectors
    Y = Xc @ Up #project data (N, n)
    return Y, Up, mew

print("running PCA to 10 dims...")
X_train_pca, Up, mew_train = pca(X_train, n=10)
# project test set using same PCA matrix
X_test_pca = (X_test - mew_train) @ Up # (N_test, 10) 
classes = numpy.array([0, 1, 2]) #target matrix 

def target(y):
    Y = numpy.zeros((len(y), 3)) #all zeros
    for i, c in enumerate(classes):
        Y[:, i] = (y == c).astype(float)
    return Y #(N, 3)
Y_train = target(y_train) #making target vector for 3 classes (N, 3)
Y_test  = target(y_test)

#80% train, 20% val
rng = numpy.random.RandomState(67) #fix seed
idx = rng.permutation(len(X_train_pca))
split = int(0.8 * len(X_train_pca))
tr, va = idx[:split], idx[split:]

Xtr, Ytr = X_train_pca[tr], Y_train[tr] #training
Xva, Yva = X_train_pca[va], Y_train[va] #validation

def add1s(X): #add col1 for constant
    return numpy.hstack([X, numpy.ones((len(X), 1))]) #(N, 11)

Xtr_b = add1s(Xtr)
Xva_b = add1s(Xva)
Xte_b = add1s(X_test_pca)

#ridge regression
def ridge_train(X, Y, lam):# W = (X^T X + lam*I)^-1 X^T Y
    n = X.shape[1]
    A = X.T @ X + lam * numpy.eye(n) #add lambda to diagonal
    W = numpy.linalg.solve(A, X.T @ Y) #solve instead of invert
    return W #shape (11, 3)

def mse(x,y):
    return numpy.mean((x-y) ** 2)

lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100] #7 values of lambda

ridge_tr_mse, ridge_va_mse, ridge_Ws = [], [], []
for lam in lambdas:
    W = ridge_train(Xtr_b, Ytr, lam) #train
    ridge_Ws.append(W)
    ridge_tr_mse.append(mse(Xtr_b @ W, Ytr)) #train error
    ridge_va_mse.append(mse(Xva_b @ W, Yva)) #val error

best_ridge_idx = int(numpy.argmin(ridge_va_mse)) # index of best lambda
best_lam_ridge = lambdas[best_ridge_idx]
print(f"best lambda (ridge): {best_lam_ridge}")

#lasso regression 
lasso_tr_mse, lasso_va_mse, lasso_Ws, lasso_nnz = [], [], [], []
for l in lambdas:
    W_cols = []
    for k in range(3):  # one lasso model per class
        model = Lasso(alpha=l, max_iter=10000, fit_intercept=False)
        model.fit(Xtr_b, Ytr[:, k]) # fit for class k target
        W_cols.append(model.coef_)
    W = numpy.column_stack(W_cols) # combine 3 class weight vectors -> (11,3)
    lasso_Ws.append(W)
    lasso_tr_mse.append(mse(Xtr_b @ W, Ytr))
    lasso_va_mse.append(mse(Xva_b @ W, Yva))
    lasso_nnz.append(numpy.sum(numpy.any(W[:-1] != 0, axis=1)))  # count non-zero feature rows (exclude bias)

best_lasso_idx = int(numpy.argmin(lasso_va_mse))
best_lam_lasso = lambdas[best_lasso_idx]
print(f"best lambda (lasso): {best_lam_lasso}")

#plot for train and val mse vs lambda for ridge and lasso  
log_lam = numpy.log10(lambdas)  #log scale for x axis
fig, axes = mtplot.subplots(1, 2, figsize=(12, 4))
for ax, tr_mse, va_mse, title in zip(axes,[ridge_tr_mse, lasso_tr_mse],[ridge_va_mse, lasso_va_mse],["Ridge Regression", "Lasso Regression"]):
    ax.plot(log_lam, tr_mse, "o-", label="Train MSE")
    ax.plot(log_lam, va_mse, "s-", label="Validation MSE")
    ax.set_xlabel("log10(lambda)")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
mtplot.tight_layout()
mtplot.show()

#plot for lasso non zero coefficients vs lambda
mtplot.figure(figsize=(6, 4))
mtplot.plot(log_lam, lasso_nnz, "o-")
mtplot.xlabel("log10(lambda)")
mtplot.ylabel("number of non-zero features")
mtplot.title("Lasso: non-zero coefficients vs lambda")
mtplot.tight_layout()
mtplot.show()

#plot forregularisation paths
#we look at class 1 weights across all lambdas
fig, axes = mtplot.subplots(1, 2, figsize=(14, 5))
for ax, Ws, title in zip(axes, [ridge_Ws, lasso_Ws], ["Ridge", "Lasso"]):
    coefs = numpy.array([W[:-1, 1] for W in Ws])  # shape (7, 10) exclude bias, class 1
    for feat in range(coefs.shape[1]):
        ax.plot(log_lam, coefs[:, feat], alpha=0.7, linewidth=1) #one line per feature
    ax.set_xlabel("log10(lambda)")
    ax.set_ylabel("coefficient value")
    ax.set_title(f"{title} regularisation path (class 1)")
mtplot.tight_layout()
mtplot.show()

#vary PCA dims plot MSE vs model complexity 
p_values = [2, 5, 10, 20, 30]
complexity_tr_mse, complexity_va_mse = [], []

for p_val in p_values:
    # refit PCA for each p value using training data
    X_tr_p, Up_p, mew_p = pca(X_train, n=p_val)
    Xtr_p = X_tr_p[tr] #training  
    Xva_p = X_tr_p[va] #validation  
    Xtr_pb = add1s(Xtr_p)
    Xva_pb = add1s(Xva_p)
    W_r = ridge_train(Xtr_pb, Ytr, best_lam_ridge)   # train ridge with best lambda
    complexity_tr_mse.append(mse(Xtr_pb @ W_r, Ytr))
    complexity_va_mse.append(mse(Xva_pb @ W_r, Yva))

mtplot.figure(figsize=(6, 4))
mtplot.plot(p_values, complexity_tr_mse, "o-", label="Train MSE")
mtplot.plot(p_values, complexity_va_mse, "s-", label="Validation MSE")
mtplot.xlabel("PCA dimensions (p)")
mtplot.ylabel("MSE")
mtplot.title(f"Ridge MSE vs model complexity (lambda={best_lam_ridge})")
mtplot.legend()
mtplot.tight_layout()
mtplot.show()

#test accuracy for best ridge and lasso models
def classify(W, Xb):
    scores = Xb @ W # (N, 3) score for each class
    return classes[numpy.argmax(scores, axis=1)]  # pick class with highest score

W_best_ridge = ridge_Ws[best_ridge_idx]
ridge_preds = classify(W_best_ridge, Xte_b)
ridge_acc = numpy.mean(ridge_preds == y_test)
print(f"Ridge test accuracy (best acc lambda={best_lam_ridge}): {ridge_acc:.2f}")

W_best_lasso = lasso_Ws[best_lasso_idx]
lasso_preds = classify(W_best_lasso, Xte_b)
lasso_acc = numpy.mean(lasso_preds == y_test)
print(f"Lasso test accuracy (best acc lambda={best_lam_lasso}): {lasso_acc:.2f}")