import numpy as np
import pandas as pd
class NeuralNetwork:
    def __init__(self, learning_rate):
        """4 weight ve 1 bias değerini random olarak atanır"""
        self.weights = np.array([np.random.randn(), np.random.randn(),np.random.randn(),np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate #learning_rate=0.01

    def leaky_Relu(self,x):
        """aktivasyon fonksiyonu"""
        return np.where(x < 0, x * 0.01, x)

    def leakyreluprime(self,x):
     #aktivasyon fonksiyonunun türevi
        if x > 0:
            return 1
        else:
            return 0.01

    def predict(self, input_vector):
        """forward propagation olarak predict fonksiyonu weight ve bias değerleri ile y=w.x+b yi oluşturur.weight
        değerleri birden çok olduğu için vektör çarpımıkullanılır.Sonra oluşan dizinin aktivasyon fonksiyonuna sokulur."""
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self.leaky_Relu(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        #predict fonksiyonunda işlemler tekrar yapılır
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self.leaky_Relu(layer_1)
        prediction = layer_2
        #backward propagaiton
        """ilk önce toplam loss daki fonksiyonun türevinin değeri hesaplanır(∂L/∂y):"""
        derror_dprediction = (prediction - target)
        """sonra y=w.x+b değerine ulaşmak için aktivasyonun türevi alınır(∂y/∂x):"""
        dprediction_dlayer1 = self.leakyreluprime(layer_1)
        """denkleme ulaştıktan sonra(layer1) weight ve bias değelerine göre türevi alınır """
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)
        """basamak basamak yapılan işlemler çarpılır(∂L/∂y.∂y/∂x.∂x/∂b):"""
        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        """(∂L/∂y.∂y/∂x.∂x/∂w):"""
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        """bias ve weight değerleri güncellenir (wnew=wnew-learning_rate*wold)"""
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )


            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)
                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors
