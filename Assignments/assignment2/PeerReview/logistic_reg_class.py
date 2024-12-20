def logistic(X):
    """Function to apply sigmoid function to X"""
    return 1/(1+np.exp(-X))
 
class NumpyLogReg(NumpyClassifier):
    """Logistic regression classifier"""
    
    def __init__(self, bias=-1):
        self.bias=bias
    
    def forward(self, X):
        return logistic(X @ self.weights)
        
    def fit(self, X_train, t_train, lr = 0.1, epochs=10, X_val = None, t_val = None, tol = 0.03, n_epochs_no_update = 5):
        """X_train is a NxM matrix, N data points, m features
        t_train is avector of length N,
        the targets values for the training data"""
    
        X_train_unbiased = X_train #Used in accuracy calculation, because self.predict would add additional bias 
        if self.bias:
                X_train = add_bias(X_train, self.bias)
                
        (N, M) = X_train.shape
        self.weights = weights = np.zeros(M)
        self.epoch_stopped = epochs
        
        self.accuracies = [] #Stores training accuracies
        self.loss_history = [] #Stores training loss
        
        #If validation set is inlcuded, store validation accuracies and loss.
        self.val_given = False
        if X_val is not None:
            self.val_given = True
            self.accuracies_val = [] #Stores validation accuracies
            self.loss_history_val = [] #Stores validation loss
            
            if (self.bias):
                X_val_unbiased = X_val
                X_val = add_bias(X_val, self.bias)
        
            
        
        no_update_count = 0
        for epoch in range(epochs):
            no_update_count += 1
            
            #Calculate and store loss for current weights
            loss = -(1/N) * np.sum(t_train * np.log(self.forward(X_train)) + (1 - t_train) * np.log(1 - self.forward(X_train)))
            self.loss_history.append(loss)
            
            #Calculate and store accuracies for current weights
            self.accuracies.append(accuracy(self.predict(X_train_unbiased), t_train))
            
            
            #If valuation set is given, calculate loss and accuracies for those as well.
            if (self.val_given):
                self.loss_history_val.append(-(1/N) * np.sum(t_val * np.log(self.forward(X_val)) + (1 - t_val) * np.log(1 - self.forward(X_val))))
                self.accuracies_val.append(accuracy(self.predict(X_val_unbiased), t_val))
            
            #Check if the loss has improved more than 'tol', assume 'n_epochs_no_update' > 1 
            if (no_update_count >= n_epochs_no_update):
                if ( abs(self.loss_history[-1] - self.loss_history[-no_update_count]) < tol):
                    self.epoch_stopped = epoch #Epoch were fitting stopped early.
                    return
                else:
                    no_update_count = 0
            weights -= lr / N * X_train.T @ (self.forward(X_train) - t_train) # Update the weight with logistic regression.
        
        
    def predict(self, X, threshold=0.5):
        """X is a KxM matrix for some K>=1
        predict the value for each point in X"""
        
        if self.bias:
            X = add_bias(X, self.bias)
            
        ys = self.forward(X)
        return ys > threshold
    
    def predict_probability(self, X):
        """
        X is a NxM matrix, instead of returning which makes the threshold,
        just return the raw probabilities.
        """
        if self.bias:
            X = add_bias(X, self.bias)
        
        return self.forward(X)