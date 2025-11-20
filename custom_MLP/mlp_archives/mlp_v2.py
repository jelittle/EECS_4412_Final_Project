
import numpy as np
"""sources:
[1] http://neuralnetworksanddeeplearning.com/
We used this as a base, but most of it is rewritten from v0
[2] Adam: a method for Stochastic Optimization, Diederik P. Kingma, Jimmy Ba, ICRL 2015

gradient rewrite
"""
class mlp:
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

 
        self.__fit=False

    def __forward(self,X):
        """forward helper for back prop and feedforward"""
        if X.ndim==3: #shape is (samples, features, 1), reshape to (features, samples)
            X=X.squeeze(-1).T
       
        assert  X.shape[0] == self.__n_features, "input features do not match model input size"
        assert X.ndim==2

        activations= [X] #keep all activations for backprop
        zs=[] #pre activation vectors
        for b,w in zip(self.biases,self.weights):
     
            z = np.dot(w, activations[-1])+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        return activations,zs


    
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
        self.__n_features= input_shape[1]
        X = X.reshape(X.shape[0], X.shape[1], 1) 
        y = y.reshape(y.shape[0], 1, 1)

        if X_test is not None:
           assert input_shape[1] == X_test.shape[1], "number of features in train and test must match"
           assert y_test.shape[0] == X_test.shape[0], "features in test input and target must match"
           X_test= X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
           y_test= y_test.reshape(y_test.shape[0], 1, 1)


        #append input output layers to layers list, initialize weights and biases
        layers=self.__params["layers"]
        layers = [X.shape[1]] + layers + [y.shape[1]]
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
    
        output,_ = self.__forward(X)
        output=output[-1]
        predictions = (output >= 0.5).astype(int).flatten()
     
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
                # if X_batch.ndim==3:
                #     X_batch= X_batch.squeeze(-1).T
                # if y_batch.ndim==3:
                #     y_batch= y_batch.squeeze(-1).T
                # assert X_batch.ndim == 2 and y_batch.ndim == 2 
      
                self.update_mini_batch(X_batch,y_batch, lr)

            # training time evaluation
            if test_data:
                n_test = test_data[0].shape[0]
      
                if self.__params["verbose"]:
                    print(f"Epoch{j}: {self.evaluate(test_data)}/{n_test}")
            else:
                print(f"Epoch{j} complete")
    def update_mini_batch(self,X,y,lr):
        """compute gradient for each example in mini batch and update weights and biases"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        X_batch = X.copy()
        y_batch = y.copy()
        if X_batch.ndim==3:
            X_batch= X_batch.squeeze(-1).T
        if y_batch.ndim==3:
            y_batch= y_batch.squeeze(-1).T
        assert X_batch.ndim == 2 and y_batch.ndim == 2 
    
        nabla_b,nabla_w=self.backprop(X_batch,y_batch)
        self.weights = [w-lr*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b-lr*nb for b,nb in zip(self.biases,nabla_b)]
    

        # exit(1)
        #temporarily add a dim 1 to x and y to match backprop input
        # X = X.reshape(X.shape[0], X.shape[1], 1)
        # y = y.reshape(y.shape[0], y.shape[1], 1)
        # mini_batch=list(zip([x for x in X], [target for target in y])) 
        # for x,y in mini_batch:
        #     delta_nabla_b, delta_nabla_w = self.backprop(x,y)
        #     nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
        #     nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        # self.weights = [w-(lr/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
        # self.biases = [b-(lr/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]
        # exit(1)
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
            sigmoid_prime(zs[-1])
        
        #divide by batch size to get average gradient
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)/batch_size
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) /batch_size
 
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
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True) / batch_size
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) / batch_size

        return (nabla_b, nabla_w)
    def evaluate(self, test_data):
        """Evaluate accuracy for binary classification.
        Assumes output neuron uses sigmoid activation and threshold 0.5."""

        x=test_data[0]

        #feed forward then get last activation
   
        output,_ = self.__forward(x)
        output=output[-1]

        predictions = (output >= 0.5).astype(int).flatten()
   
        y=test_data[1].flatten()
        accuracy = np.sum(predictions == y)
        return accuracy
       
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))