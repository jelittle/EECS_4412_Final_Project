
import numpy as np
"""sources:
[1] http://neuralnetworksanddeeplearning.com/
We used this as a base, but most of it is rewritten from v0
https://arxiv.org/pdf/1803.08375 relu activation 
Adam: a method for Stochastic Optimization, Diederik P. Kingma, Jimmy Ba, ICRL 2015


"""
class mlpClassifier:

    def __init__(self,layers,activation="relu",solver="adam",
                 learning_rate=0.001,batch_size=32,epochs=100,random_state=None,
                 verbose=False):
        self.__params = {
            "solver": solver,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "activation": activation,
            "layers": layers,
            "batch_size": batch_size,
            "verbose": verbose
        }

        if random_state is not None:
            np.random.seed(random_state)
        self.activation_func=get_activation_func(activation)
 
        self.__fit=False
    def __forward(self, X):
        """forward helper for back prop and feedforward"""
        if X.ndim == 3:  # shape is (samples, features, 1), reshape to (features, samples)
            X = X.squeeze(-1).T

        assert X.shape[0] == self.__n_features, "input features do not match model input size"
        assert X.ndim == 2

        activations = [X]  # keep all activations for backprop
        zs = []  # pre activation vectors
        num_layers = len(self.biases)
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            if i == num_layers - 1:
                activation = softmax(z) if self.__n_classes > 1 else sigmoid(z)
            else:
                activation = self.activation_func(z)
            activations.append(activation)
        return activations, zs



    
    def fit(self,X, y, X_test=None, y_test=None):
        """sets up firs tna dlast layer based on input and output data, and trains the network using SGD
        aim to match sklearn and pyTorch conventions
        input shape: (n_samples, n_features)
        target shape: (n_samples,) (model only supports binary classification currently)"""
        #-----------------make sure input makes sense
        input_shape= X.shape
        target_shape= y.shape
        assert len(input_shape) ==2, "input data must be 2D array"
        assert len(target_shape) ==1, "target data must be 1D array"
        assert input_shape[0] == target_shape[0], "number of samples in input and target must match"
        
        n_unique = len(np.unique(y))
        self.__task = "binary" if n_unique == 2 else "multiclass"
        self.__n_classes = 1 if self.__task == "binary" else n_unique
        
        # handle multiclassing
        if self.__task == "multiclass":
            y_onehot = np.zeros((y.shape[0], self.__n_classes))
            y_onehot[np.arange(y.shape[0]), y.astype(int)] = 1
            y = y_onehot
            if y_test is not None:
                y_test_onehot = np.zeros((y_test.shape[0], self.__n_classes))
                y_test_onehot[np.arange(y_test.shape[0]), y_test.astype(int)] = 1
                y_test = y_test_onehot
        
        
        
        self.__n_features= input_shape[1]
        X = X.reshape(X.shape[0], X.shape[1], 1) 
        y = y.reshape(y.shape[0], -1, 1)

        if X_test is not None:
           assert input_shape[1] == X_test.shape[1], "number of features in train and test must match"
           assert y_test.shape[0] == X_test.shape[0], "features in test input and target must match"
           X_test= X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
           y_test= y_test.reshape(y_test.shape[0], -1, 1)


        #append input output layers to layers list, initialize weights and biases
        layers=self.__params["layers"]
        layers = [X.shape[1]] + layers + [self.__n_classes]
        self.num_layers = len(layers)
         
        self.weights = [np.random.randn(y,x) for x,y  in zip(layers[:-1],layers[1:])]
        self.biases = [np.random.randn(y,1) for y in layers[1:]]

        train_data = (X, y)
        test_data = (X_test, y_test) if X_test is not None else None  
  


        epochs = self.__params["epochs"]
        learning_rate = self.__params["learning_rate"]
        batch_size = self.__params["batch_size"]
        # Call SGD
        self.SGD(train_data, epochs, batch_size, learning_rate, test_data)
        self.__fit=True


    def predict(self,X):
        """predict class labels for input data X
        input shape: (n_samples, n_features)
        output shape: (n_samples,)"""
        assert self.__fit, "model must be fit before prediction"
        input_shape= X.shape
        assert len(input_shape) ==2, "input data must be 2D array"
        X = X.reshape(X.shape[0], X.shape[1], 1)
        output, _ = self.__forward(X)
        output = output[-1]
        
        if self.__n_classes == 1:
            predictions = (output >= 0.5).astype(int).flatten()
        else:
            predictions = np.argmax(output, axis=0)  # argmax over classes
        
        return predictions
    def SGD(self, training_data, epochs, mini_batch_size, lr, test_data):
        """training_data is a tuple (X, y) of numpy arrays """
        #prep data
        X, y = training_data
        n = X.shape[0]

           

        
        for j in range(epochs):
            # Shuffle the data
            indices = np.arange(n)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            # Create mini-batches
            for k in range(0, n, mini_batch_size):
                X_batch = X_shuffled[k:k+mini_batch_size]
                y_batch = y_shuffled[k:k+mini_batch_size]
                if X_batch.ndim==3:
                    X_batch= X_batch.squeeze(-1).T
                if y_batch.ndim==3:
                    y_batch= y_batch.squeeze(-1).T
                assert X_batch.ndim == 2 and y_batch.ndim == 2 
      
                self.update_mini_batch(X_batch,y_batch, lr)
            if self.__params["verbose"]:
       
                # training time evaluation
                if test_data:
                    n_test = test_data[0].shape[0]
                    print(f"Epoch{j}: {self.evaluate(test_data)}/{n_test}")

                else:
                    print(f"Epoch{j} complete")
    def update_mini_batch(self,X,y,lr):
        """compute gradient for each example in mini batch and update weights and biases"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
    
        nabla_b,nabla_w=self.backprop(X,y)
        self.weights = [w-lr*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b-lr*nb for b,nb in zip(self.biases,nabla_b)]
    

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""


        batch_size=x.shape[1]
 
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
  
        activations, zs=self.__forward(x)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            activation_prime(zs[-1],sigmoid) #always use sigmoid at output
        
        #divide by batch size to get average gradient
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)/batch_size
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) /batch_size
 
        #backpropagate through layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp =  activation_prime(z,self.activation_func)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True) / batch_size
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) / batch_size

        return (nabla_b, nabla_w)
    def evaluate(self, test_data):
        x = test_data[0]
        output, _ = self.__forward(x)
        output = output[-1]
        
        if self.__n_classes == 1:
            predictions = (output >= 0.5).astype(int).flatten()
            y = test_data[1].flatten()
        else:
            predictions = np.argmax(output, axis=0)
            y = np.argmax(test_data[1].squeeze(-1), axis=1)
        
        accuracy = np.sum(predictions == y)
        return accuracy
       
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
def get_activation_func( name):
    activation_funcs = {
        "sigmoid": sigmoid,
        "relu": relu,
    }
    assert name in activation_funcs, f"Unsupported activation function: {name}"
    return activation_funcs[name]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)
def softmax(z):
    """Numerically stable softmax. z shape: (n_classes, batch_size)"""
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)
def softmax_prime(z):
    """Softmax derivative for MSE loss"""
    s = softmax(z)
    return s * (1 - s)  # simplified diagonal approximation

activation_derivatives = {
    sigmoid: sigmoid_prime,
    relu: relu_prime,
    softmax: softmax_prime,
}

def activation_prime(x, func):
    if func in activation_derivatives:
        return activation_derivatives[func](x)
    else:
        raise ValueError("Unsupported activation function")