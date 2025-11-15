
import numpy as np
"""sources:
http://neuralnetworksanddeeplearning.com/
currently, backprop, evaluate, and cost_derivate are directly taken from the above source
"""

class network:
    def __init__(self,layers,activation="relu",solver="adam",learning_rate=0.001,epochs=100,random_state=None):
        self.params = {
            "solver": solver,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "activation": activation,
            "layers": layers
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


# class mlp:
#     def __init__(self, layers,activation="relu",solver="adam",learning_rate=0.001,epochs=100,random_state=None):
#         self.params = {
#             "solver": solver,
#             "learning_rate": learning_rate,
#             "epochs": epochs,
#             "activation": activation,
#             "layers": layers
#         }
#         if random_state is not None:
#             np.random.seed(random_state)

#         self.weights = [np.random.randn(y,1) for y in layers for y in layers[1:]]
#         print(len(self.weights))
#         self.biases = [np.random.randn(y,1) for y in layers for y in layers[1:]]

#     def fit(self, X, y):
#         pass
#     def predict(self, X):
#         pass


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))