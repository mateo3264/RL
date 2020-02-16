import numpy as np
def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class NN:
    def __init__(self,layers,learning_rate,factor_weights):
        self.layers = layers
        self.num_layers = len(layers)

        self.weights = [np.random.random((y,x))*factor_weights for x,y
                        in zip(layers[:-1],layers[1:])]
        self.biases = [np.zeros((x,1)) for x in layers[1:]]
        self.learning_rate = learning_rate
        self.error = float('inf')

    def backpropagation(self,x,y):
        a = x
        activations = [a]
        zs = []

        for l in range(self.num_layers-1):
            z = np.dot(self.weights[l],a) + self.biases[l]
            a = sigmoid(z)
            activations.append(a)
            zs.append(z)
        self.error = a-y
        delta = self.error*sigmoid_prime(z)
        da = np.dot(self.weights[-1].T,delta)
        dw = np.dot(delta,activations[-2].T)
        db = delta
        self.weights[-1] -=self.learning_rate*dw
        self.biases[-1] -=self.learning_rate*db

        for l in range(2,self.num_layers):
            a_actual = activations[-l]
            z_actual = zs[-l]
            delta = da*sigmoid_prime(z_actual)
            dw = np.dot(delta,activations[-l-1].T)
            db = delta
            self.weights[-l] -=self.learning_rate*dw
            self.biases[-l] -=self.learning_rate*db
            da = np.dot(self.weights[-l].T,delta)

    def forward_propagation(self,x):
        activation = x
        for l in range(self.num_layers-1):
            activation = sigmoid(np.dot(self.weights[l],activation) + self.biases[l])
        activation[activation>0.5] = 1
        activation[activation<=0.5] = 0
        return activation
