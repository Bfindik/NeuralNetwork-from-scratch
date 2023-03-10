import numpy as np
class neural_network:
    def __init__(self, learning_rate):
        self.w1 = np.random.rand(3, 4) - 0.5
        self.b1 = np.random.rand() 
        self.w2 = np.random.rand(1, 3) - 0.5
        self.b2 = np.random.rand() - 0.5
        self.learning_rate = learning_rate
    def ReLU_deriv(self,Z):
        return Z > 0

    def ReLU(self,Z):
        return np.maximum(Z, 0)
    """def softmax(self,Z):
         A = np.exp(Z) / sum(np.exp(Z))
         return A"""
    def forward_prop(self,input_vector):
        z1 = np.dot(self.w1, input_vector) + self.b1
        a1 = self.ReLU(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = self.ReLU(z2)
        prediction = a2
        return prediction

    def train(self, X_train, y_train, epochs):
       for epoch in range(epochs):
         for i in range(len(X_train)):
            # Forward propagation
            input_vector = X_train[i]
            actual_output = y_train[i]
            z1 = np.dot(self.w1, input_vector) + self.b1
            a1 = self.ReLU(z1)
            z2 = np.dot(self.w2, a1) + self.b2
            predicted_output = self.ReLU(z2)
            
            # Backpropagation
            output_error = predicted_output - actual_output
            #output_error = output_error.reshape(1,1)
            z1_error = np.dot(self.w2.T, output_error) * self.ReLU_deriv(z1)
            w2_error = np.dot(a1.reshape(3,1), output_error.reshape(1,1))
            w1_error = np.dot(input_vector.reshape(4,1), z1_error.reshape(1,3))
            # Update weights
            self.w2 -= self.learning_rate * w2_error.T
            self.b2 -= self.learning_rate * output_error
            self.w1 -= self.learning_rate * w1_error.T
            self.b1 -= self.learning_rate * z1_error.sum()
      
           