import numpy as np
import matplotlib.pyplot as mtplot
import os
from urllib.request import urlretrieve
from sklearn.manifold import TSNE

def load_mnist():
    filename = "mnist.npz"
    if not os.path.exists(filename):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz" #download dataset if it doesnt exist
        urlretrieve(url, filename)

    with np.load(filename) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

print("loading data...")
(x_train_, y_train_), (x_test_, y_test_) = load_mnist() #load data using defined func

# preprocess data func
def preprocess_data(x_data, y_data):
    filter = np.isin(y_data, [0, 1, 2])
    x_filter = x_data[filter]
    y_filter = y_data[filter]
    
    x, y = [], []
    rng = np.random.RandomState(67) #fixing seed
    
    for i in [0, 1, 2]:
        all_ind = np.where(y_filter == i)[0] #get all indices for class i
        choose = rng.choice(all_ind, 100, replace=False) #100 samples for class i
        x.append(x_filter[choose])
        y.append(y_filter[choose]) 
    X_combined = np.vstack(x)
    y_combined = np.hstack(y)
    X_flat = X_combined.reshape(X_combined.shape[0], -1) #flatten image to 784d vector
    X_norm = X_flat.astype(np.float64) / 255.0 #normalize to floats
    return X_norm, y_combined

X_train, y_train = preprocess_data(x_train_, y_train_)
X_test, y_test = preprocess_data(x_test_, y_test_)

# mle est for mean and cov 
def compute_mle(X, y):
    params = {}
    classes = np.unique(y) #[0,1,2]
    for i in classes:
        X_i = X[y == i]
        N = X_i.shape[0] #N=100
        mu = np.mean(X_i, axis=0) #mean of each col
        diff = X_i - mu 
        sigma = np.dot(diff.T, diff) / N #cov matrix
        sigma += np.eye(sigma.shape[0]) * 0.0001 #making sure mat is invertible
        
        # calculate inv and logdet
        sign, logdet = np.linalg.slogdet(sigma)
        sigma_inv = np.linalg.inv(sigma)
        
        params[i] = {'mu': mu,'sigma': sigma,'sigma_inv':sigma_inv,'logdet': logdet}
    return params

model_params = compute_mle(X_train, y_train)

def predict(X, params, method):
    preds = []
    scores_list = []
    
    lda_inv = None
    lda_logdet = None
    
    if method == 'LDA': #precalculate common sigma inv and logdet for LDA
        all_sigmas = [p['sigma'] for p in params.values()]
        common_sigma = np.mean(all_sigmas, axis=0) #one cov matrix for all classes
        sign, lda_logdet = np.linalg.slogdet(common_sigma) #log sigma
        lda_inv = np.linalg.inv(common_sigma) #sigma inv
    
    for x in X:
        class_scores = {}
        for i in params:
            mu = params[i]['mu']
            
            if method == 'QDA':
                sigma_inv = params[i]['sigma_inv']
                logdet = params[i]['logdet']
            else: #method = LDA
                sigma_inv = lda_inv
                logdet = lda_logdet
            
            d = len(mu) #dim = 784
            diff = x - mu
            expo = diff @ sigma_inv @ diff.T #(x-mu) * sigmainv * (x-mu)^T
            log_p = -0.5 * (d * np.log(2 * np.pi) + logdet + expo)
            
            class_scores[i] = log_p
        
        c = max(class_scores, key=class_scores.get)
        preds.append(c)
        scores_list.append(class_scores)
        
    return np.array(preds), scores_list

print("\rfinal results...")
print("QDA...")
qda_pred, qda_scores = predict(X_test, model_params, 'QDA')
qda_acc = np.mean(qda_pred == y_test)
print(f"QDA Accuracy: {qda_acc:.4f} ({qda_acc*100:.2f}%)")

print("LDA...")
lda_pred, lda_scores = predict(X_test, model_params, 'LDA')
lda_acc = np.mean(lda_pred == y_test)
print(f"LDA Accuracy: {lda_acc:.4f} ({lda_acc*100:.2f}%)")

print(f"True Class: {y_test[0]}")
print("\nQDA disc values for sample 0 --")
for c, val in qda_scores[0].items():
    print(f"Class {c}: {val:.4f}")
print("\nLDA disc values for sample 0 --")
for c, val in lda_scores[0].items():
    print(f"Class {c}: {val:.4f}")

def plot_tsne(X, y, title): #plotting func for t-SNE
    print(f"\nRunning t-SNE for {title}...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)
    mtplot.figure(figsize=(10,7))
    scatter = mtplot.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='viridis', alpha=0.7)
    handles, _ = scatter.legend_elements(prop="colors")    
    mtplot.xlabel("t-SNE Dimension 1")
    mtplot.ylabel("t-SNE Dimension 2")
    mtplot.legend(handles, ["0", "1", "2"], title="Digit Class")
    mtplot.title(f"{title} (t-SNE)")
    mtplot.show()

# run for both sets
plot_tsne(X_train, y_train, "Train Set")
plot_tsne(X_test, y_test, "Test Set")
