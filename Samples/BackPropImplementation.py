import numpy as np
import matplotlib.pyplot as plt


def generate_dataset(sample=500,dimension=5,classes=3):
    X = np.random.randn(sample,dimension)
    y = np.zeros(sample)

    #divide sample into equal class sizes
    for i in range(classes):
        size = ((i+1)*sample//classes-i*sample//classes)
        X[i*sample//classes:(i+1)*sample//classes] += np.random.randint(-5,5,dimension)
        y[i*sample//classes:(i+1)*sample//classes] = i

    return X,y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def softmax(z):
    return np.exp(z)/np.exp(z).sum(axis=1)[:,np.newaxis]

def logistic_regression(X,y):
    #add bias term
    X_new = np.concatenate([np.zeros((X.shape[0],1)), X], axis=1)
    y = y[:,np.newaxis]

    #init weights
    W = np.random.randn(X_new.shape[1],1)

    learnrate = 1e-4
    losses = []
    for epoch in range(1000):
        #forward propogate
        y_hat = sigmoid(X_new.dot(W))
        dW = X_new.T.dot(y*(1-y_hat))

        W += learnrate*dW
        loss = -(y*np.log(y_hat)).sum()
        losses.append(loss)

    print(y_hat.round())

    fig,axes = plt.subplots(ncols=3,figsize=(12,8))
    ax = axes.ravel()
    ax[0].plot(losses,"r*")
    ax[0].set_title("Learning Rate")
    ax[1].scatter(X[:,0],X[:,1],c=y)
    ax[1].set_title("Truth")
    ax[2].scatter(X[:,0],X[:,1],c=y_hat.round())
    ax[2].set_title("Prediction")
    plt.show()


def plot_data(X,y):
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.show()
    #theta = fit_multiclass(X,y)
    #print(theta)

if __name__=="__main__":
    X,y = generate_dataset(sample=100,dimension=4,classes=2)
    #plot_data(X,y)
    logistic_regression(X,y)
