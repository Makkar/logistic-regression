"""
Logistic Regression
Author: Aditya Makkar
"""

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression():
    
    def __init__(self, verbose=False):
        self.w = None
        self.b = None
        self.X_train = None
        self.Y_train = None
        self.verbose = verbose
    
    def sigmoid(z):
        """
        Returns the sigmoid of z.
        """
        s = 1 / (1 + np.exp(-z))
        return s
    
    def _initialize_params(self, dim):
        """
        Initialize w and b to zeros.
        """
        self.w = np.zeros((dim, 1))
        self.b = 0
    
    def _propagate(self):
        """
        We do one step of forward propagation and one step of backward propagation
        to get the cost and gradients.
        """
        #assert (self.X_train != None and self.Y_train != None), 'Error: No training data'
        m = self.X_train.shape[1] # Number of training datapoints
        # Forward propagation
        A = LogisticRegression.sigmoid(np.dot(self.w.T, self.X_train) + self.b) # activation
        cost = (-1 / m) * np.sum(self.Y_train * np.log(A) + (1 - self.Y_train) * np.log(1 - A)) # cost
        cost = np.squeeze(cost)
        # Backward propagation
        dw = (1 / m) * np.dot(self.X_train, (A - self.Y_train).T) # Derivative of cost function wrt w
        db = (1 / m) * np.sum(A - self.Y_train) # Derivative of cost function wrt b
        grads = {'dw': dw, 'db': db}
        return grads, cost
    
    def _optimize(self, num_iter, lr, verbose=False):
        """
        Optimize w and b by gradient descent.
        """
        costs = []
        for i in range(num_iter):
            grads, cost = self._propagate()
            costs.append(cost)
            dw = grads['dw']
            db = grads['db']
            # Gradient descent
            self.w = self.w - lr * dw
            self.b = self.b - lr * db
            if (verbose or self.verbose) and i % 100 == 0:
                print("Cost after iteration {0}: {1}".format(i, cost))
        return costs
    
    def predict(self, X):
        """
        Using the trained model, predict on X.
        """
        m = X.shape[0]
        X = X.reshape(m, -1).T
        # Predict the probabilities
        A = LogisticRegression.sigmoid(np.dot(self.w.T, X) + self.b)
        Y = np.where(A > 0.5, 1, 0)
        return Y
    
    def train(self, X, Y, num_iter=2000, lr=0.01, verbose=False):
        """
        Trains a logistic regression model using X and Y.
        
        X - A numpy array containing features for all training examples.
        X.shape[0] should be number of training examples. For example, for 100 
        training images of shape 200 x 200 pixels with 3 RGB components, 
        X.shape = (100, 200, 200, 3).
        
        Y - Training labels. Shape should be (m,), (m, 1), or (1, m).
        """
        m = X.shape[0] # Number of training examples
        X = X.reshape(m , -1).T
        Y = Y.reshape(1, m)
        n = X.shape[0]
        self._initialize_params(n)
        self.X_train = X
        self.Y_train = Y
        costs = self._optimize(num_iter, lr, verbose)
        if verbose or self.verbose:
            plt.plot(costs)
            plt.ylabel('cost')
            plt.xlabel('iterations')
            plt.title("Learning rate =" + str(lr))
            plt.show()

if __name__ == '__main__':
    """
    Sample usage.
    We will construct data such that it has three features and we want label to
    be 1 if sum of first two features > third feature, else 0.
    """
    X = np.array([[0.4, 0.5, 0.8],
                  [0.6, 0.1, 0.9],
                  [0.1, 1.4, 2.4],
                  [2.1, 2.1, 1.5]])
    Y = np.array([1, 0, 0, 1])
    log_reg = LogisticRegression(verbose=True)
    log_reg.train(X, Y, num_iter=10000, lr=0.1)
    X_test = np.array([[1.3, 2.4, 2.1],
                       [0.4, 0.9, 3.1]])
    print('Predicted labels: ', log_reg.predict(X_test))
    print('Expected labels: ', np.array([[1, 0]]))
