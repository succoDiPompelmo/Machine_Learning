import numpy as np

"""
Basic implementation of a Perceptron model, for the only purpose of testing,
I'm going to use the skilearn implentation indeed 
"""

class AdalineGD():
    '''
    Perceptron classifier

    Parameters
    ---------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over training dataset
    shuffle : bool (default: True)
        Shuffle trainig data every epoch to prevent cycles
    random_state: int
        Random number generator seed for random weight initialization

    Attributes
    ---------------
    w_ : 1d-array
        Weights after fitting
    cost_ : list
        Sum-of-squares cost function value in each epoch
    '''

    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_initialized = False
        self.random_state = random_state

    def fit(self, X, y):
        '''
        Fit trainig data

        Parameters
        ---------------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of
            features
        y : array-like, shape = [n_samples]

        Returns
        -----------
        self: object
        '''
        self._initialized_weights(X.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit trainig data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialized_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
            return self

    def net_input(self, X):
        """Calculate the net input"""
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Calculate the linear activation"""
        return X

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialized_weights(self, m):
        """Initialize weights to small random number"""
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + m)

        self.w_initialized = True

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)