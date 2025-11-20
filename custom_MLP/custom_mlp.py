
import numpy as np
"""sources:
http://neuralnetworksanddeeplearning.com/
currently, backprop, evaluate, and cost_derivate are directly taken from the above source
"""

class mlp:
    def __init__(self,layers,activation="relu",solver="adam",learning_rate=0.001,batch_size=32,epochs=100,random_state=None):
        self.__params = {
            "solver": solver,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "activation": activation,
            "layers": layers,
            "batch_size": batch_size
        }
        if random_state is not None:
            np.random.seed(random_state)
        self.num_layers=(len(layers))
 
        self.weights = [np.random.randn(y,x) for x,y  in zip(layers[:-1],layers[1:])]
  
        self.biases = [np.random.randn(y,1) for y in layers[1:]]
    def feedforward(self,X):
        for b,w in zip(self.biases,self.weights):
            X = sigmoid(np.dot(w,X)+b)
        return X
    
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
        # assert input_shape[0] == target_shape[0], "number of samples in input and target must match"
        
        X_reshaped = X.T.reshape(2, -1, 1).transpose(1, 0, 2)  
        y_reshaped = y.reshape(-1, 1, 1)  #
        X_test_reshaped=X_test.T.reshape(2,-1,1).transpose(1,0,2)
        y_test_reshaped=y_test.reshape(-1,1,1)
        assert np.allclose(X_reshaped.squeeze(), X), "X_reshaped values do not match X"
        exit(1)
        # Convert to list of tuples
        train_data = [(X_reshaped[i], y_reshaped[i]) for i in range(len(X))]
        test_data =  [(X_test_reshaped[i], y_test_reshaped[i]) for i in range(len(X_test))]
            # if test_X is not None:
        #    assert input_shape[1] == test_X.shape[1], "number of features in train and test input must match"
        #    assert test_y.shape[0] == test_X.shape[0], "number of samples in test input and test target must match"
        #    test_X= test_X.reshape(test_X.shape[0], test_X.shape[1], 1)
        #    test_y= test_y.reshape(test_y.shape[0], 1, 1)
        # # Reshape X to (samples, features, 1)
        # X = X.reshape(X.shape[0], X.shape[1], 1)
        # y = y.reshape(y.shape[0], 1, 1)
        # #-----------------end input checks

        # #add nodes to input and output layers based on data
        # n_input = input_shape[1]
        # n_output = 1  # binary classification

        # #build structure of network
        # layers=self.__params["hidden_layers"]
        # layers = [n_input] + layers + [n_output]
        # # self.num_layers = len(layers)
        # # self.weights = [np.random.randn(y,x) for x,y  in zip(layers[:-1],layers[1:])]

        # # self.biases = [np.random.randn(y,1) for y in layers[1:]]


        # #train
        # # Pair training data as list of (x, y) tuples
        # training_data = list(zip([x for x in X], [target for target in y]))
        # # handle test data
        # test_data = list(zip([x for x in test_X], [target for target in test_y])) if test_X is not None else None
        # # Set parameters for SGD

        epochs = self.__params["epochs"]
        learning_rate = self.__params["learning_rate"]
        batch_size = self.__params["batch_size"]
        # Call SGD
        self.SGD(train_data, epochs, batch_size, learning_rate, test_data)
        self.__fit=True
    def SGD(self,training_data,epochs,mini_batch_size,lr, test_data):
        """data is list of tuples (x,y)"""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)
            if test_data:
                print(f"Epoch{j}:{self.evaluate(test_data)}/{n_test}")
            else:
                print(f"Epoch{j} complete")
    def update_mini_batch(self,mini_batch,lr):
        """compute gradient for each example in mini batch and update weights and biases"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weights = [w-(lr/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b-(lr/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    def evaluate(self, test_data):
        """Evaluate accuracy for binary classification.
        Assumes output neuron uses sigmoid activation and threshold 0.5."""
        test_results = []
        for x, y in test_data:
            output = self.feedforward(x)
            prediction = int(output >= 0.5)
            # If y is one-hot or array, get scalar
            if isinstance(y, np.ndarray) and y.size == 1:
                y_val = int(y[0])
            else:
                y_val = int(y)
            test_results.append((prediction, y_val))
        return sum(int(pred == y_true) for pred, y_true in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))