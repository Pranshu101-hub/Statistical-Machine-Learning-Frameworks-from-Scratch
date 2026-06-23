import os
import numpy as np
import matplotlib.pyplot

from sklearn.datasets import fetch_openml 

# load the raw data
def load_():
    f = "sml_a2.npz"
    if not os.path.exists(f):
        print("downloading dataset...")
        mnist = fetch_openml('mnist_784', version=1, parser='liac-arff', as_frame=False)
        xall = mnist.data
        yall = mnist.target.astype(int)
        np.savez(f, x=xall, y=yall)  #save localcopy
    print("loading dataset from cache...")
    data = np.load(f)
    xall, yall = data['x'], data['y']
    return xall, yall

# preprocess data for 0,1,2 
def preprocess_(x_data, y_data):
    filt = np.isin(y_data, [0, 1, 2]) #we only need 0,1,2
    x_filtered, y_filtered = x_data[filt], y_data[filt]
    x_filtered = x_filtered / 255.0 #normalising to 0 to 1
    x_train_, x_test_, y_train_, y_test_ = [], [], [], [] #train test data and labels 
    np.random.seed(67) #fix seed
    for c in [0, 1, 2]: #selecting each class and 100 samples for train and 100 for test
        x_c = x_filtered[y_filtered == c] #all samples for c
        y_c = y_filtered[y_filtered == c] #all labels for c
        i = np.random.permutation(len(x_c)) 
        x_train_.append(x_c[i[:100]])
        y_train_.append(y_c[i[:100]])
        x_test_.append(x_c[i[100:200]])
        y_test_.append(y_c[i[100:200]])
    x_train = np.vstack(x_train_).T #stack and transpose to match (784, 300)
    y_train = np.hstack(y_train_)
    x_test = np.vstack(x_test_).T   
    y_test = np.hstack(y_test_)
    
    return x_train, y_train, x_test, y_test

x_raw, y_raw = load_()
x_train, y_train, x_test, y_test = preprocess_(x_raw, y_raw)
print(f"x_train: {x_train.shape}") # belongs to R^784, 300 
print(f"x_test: {x_test.shape}")

# dim reduction by pca
def pca(x, variance_threshold=0.75, n_components=None):
    mew = np.mean(x, axis=1, keepdims=True) #allowed to use np.mean according to comments on the assignment
    xc = x - mew #centring data
    N = x.shape[1] #no of samples
    S = (xc @ xc.T) / (N - 1) #cov of centre 
                 
    eigval, eigvec = np.linalg.eig(S)
    eigval = np.real(eigval)
    eigvec = np.real(eigvec) #get real parts to avoid numerical issues
    
    i = np.argsort(eigval)[::-1] #sort desc
    eigval = eigval[i] #sort eigval
    eigvec = eigvec[:, i] #sort eigvec, as the columns correspond to eigval and the first column is the eigenvector for the largest eigenvalue
    cumvar = np.cumsum(eigval) / np.sum(eigval) 
    
    if n_components is not None:
        k = n_components #used for 2 components case, k=2
    else:
        k = np.argmax(cumvar >= variance_threshold) + 1 #else just check for variance threshold, idx+1 gives no of comp 
    Up = eigvec[:, :k] #first k eigvec
    Y = Up.T @ xc #projecting centred data onto eigvec
    return Y, Up, mew, eigval

y_train_varth, Up_varth, mew_train, _ = pca(x_train)

# fda implementation
def fda(x, y):
    d, N = x.shape #784,300
    cs = np.unique(y) #c=0,1,2
    mewg = np.mean(x, axis=1, keepdims=True) #global mean
    Sw = np.zeros((d, d)) #scatter within 
    Sb = np.zeros((d, d)) #scatter between
    
    for c in cs:
        xc = x[:, y == c] 
        Nc = xc.shape[1] 
        mewc = np.mean(xc, axis=1, keepdims=True) 
        diff_w = xc - mewc 
        Sw += diff_w @ diff_w.T #building Sw by adding the Sc of every class

        diff_b = mewc - mewg
        Sb += Nc * (diff_b @ diff_b.T) #building Sb
        
    Sw_reg = Sw + 1e-4 * np.eye(d) #regularize Sw for det non zero 
    
    Sw_inv = np.linalg.inv(Sw_reg)
    M = Sw_inv @ Sb #Sw_inv * Sb * w = lamb * w
    
    eigval, eigvec = np.linalg.eig(M)
    eigval = np.real(eigval)
    eigvec = np.real(eigvec) 
    i = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, i]

    W = eigvec[:, :2] 
    return W

W_fda = fda(x_train, y_train) 
y_train_fda = W_fda.T @ x_train
y_test_fda = W_fda.T @ x_test

#lda and qda from A1
def fit(x, y):
    d, N = x.shape
    classes = np.unique(y)
    params = {}
    for c in classes:
        xc = x[:, y == c]
        Nc = xc.shape[1]
        muc = np.mean(xc, axis=1)
        diff = xc - muc[:, None]
        sigma_c = (diff @ diff.T) / (Nc - 1) + np.eye(d) * 1e-6
        sign, logdet = np.linalg.slogdet(sigma_c)
        params[c] = {'mu': muc,'log_prior': np.log(Nc / N),'sigma': sigma_c,'sigma_inv': np.linalg.inv(sigma_c),'logdet': logdet}
    return params

def predict(X, params, method):
    preds = []
    lda_inv = lda_logdet = None
    if method == 'LDA':
        all_sigmas = [p['sigma'] for p in params.values()]
        common_sigma = np.mean(all_sigmas, axis=0)
        _, lda_logdet = np.linalg.slogdet(common_sigma)
        lda_inv = np.linalg.inv(common_sigma)
    
    for x in X.T:  
        class_scores = {}
        for i in params:
            mu = params[i]['mu']
            log_prior = params[i]['log_prior']
            sigma_inv = lda_inv if method == 'LDA' else params[i]['sigma_inv']
            logdet = lda_logdet if method == 'LDA' else params[i]['logdet']
            diff = x - mu
            expo = diff @ sigma_inv @ diff.T
            log_like = -0.5 * (logdet + expo)  
            class_scores[i] = log_prior + log_like
        preds.append(max(class_scores, key=class_scores.get))
    return np.array(preds)

#reconstruct and mean squared error
def reconstruct(Y, Up, mew, og_X):
    y_samples = Y[:, :5] #take first 5 samples 
    X_og_samples = og_X[:, :5]
    recon = (Up @ y_samples) + mew
    total_mse = 0
    for i in range(5):
        mse = np.mean((X_og_samples[:, i] - recon[:, i])**2)
        total_mse += mse
        print(f"Sample {i+1} MSE: {mse:.7f}")   
    print("\n")
    print(f"Average MSE for 5 samples: {total_mse / 5:.7f}")
    
    return X_og_samples, recon

X_orig, recon = reconstruct(y_train_varth, Up_varth, mew_train, x_train)

#report
fig, axes = matplotlib.pyplot.subplots(2, 5)
fig.suptitle("pca reconstruction (75% var)")
for i in range(5):#plot original and reconstructed images side by side
    axes[0, i].imshow(X_orig[:, i].reshape(28, 28), cmap='gray') #show og img    
    axes[1, i].imshow(recon[:, i].reshape(28, 28), cmap='gray')#show reconstructed img and mse 

matplotlib.pyplot.tight_layout()
matplotlib.pyplot.show()

def calc_acc(x, y): #calculating accuracy 
    c = 0
    for i in range(len(y)):
        if y[i] == x[i]:  
            c += 1
    ac = c/len(y)
    return ac

def evaluate(x_tr, y_tr, x_te, y_te, title):
    params = fit(x_tr, y_tr)
    for method in ['LDA', 'QDA']:
        train_preds = predict(x_tr, params, method)
        test_preds  = predict(x_te, params, method)
        train_acc = calc_acc(train_preds, y_tr)
        test_acc  = calc_acc(test_preds, y_te)
        print("*"*50)
        print(title)
        print(f"{method}: train accuracy= {train_acc:.7f}")
        print(f"{method}: test accuracy= {test_acc:.7f}")

#execution 
evaluate(y_train_fda, y_train, y_test_fda, y_test, "fda (2d space)") #fda results

y_test_varth = Up_varth.T @ (x_test - mew_train)
evaluate(y_train_varth, y_train, y_test_varth, y_test, "pca (75% var)") #pca with variance threshold results

y_train_90,Up_90,a,a= pca(x_train, variance_threshold=0.90)
y_test_90 = Up_90.T @ (x_test - mew_train)
evaluate(y_train_90, y_train, y_test_90, y_test, "pca (90% var)") #pca with 90% variance results

y_train_pca, Up_pca,a,a = pca(x_train, n_components=2)
y_test_pca = Up_pca.T @ (x_test - mew_train)
evaluate(y_train_pca, y_train, y_test_pca, y_test, "pca (Only 2 comps)") #pca with only 2 components results

# transformed feat spaces
def plot_combine(y_fda, y_pca, labels):
    fig, axes = matplotlib.pyplot.subplots(1, 2)
    colors = ['red', 'green', 'blue']
    for c, color in zip([0, 1, 2], colors): #fda 2d projection space
        axes[0].scatter(y_fda[0, labels == c], y_fda[1, labels == c], c=color, marker='o', label=f'digit {c}', alpha=1, edgecolors='k')
    axes[0].set_title("fda 2D proj space (train data)")
    axes[0].set_xlabel("fisher comp 1")
    axes[0].set_ylabel("fisher comp 2")
    axes[0].legend()
    for c, color in zip([0, 1, 2], colors):  #pca 2d projection space
        axes[1].scatter(y_pca[0, labels == c], y_pca[1, labels == c], c=color, marker='o', label=f'digit {c}', alpha=1, edgecolors='k')
    axes[1].set_title("pca 2-comp space (train data)")
    axes[1].set_xlabel("principal comp 1")
    axes[1].set_ylabel("principal comp 2")
    axes[1].legend()
    
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()
plot_combine(y_train_fda, y_train_pca, y_train)
