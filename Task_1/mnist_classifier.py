from classifiers.random_forest import RandomForestMnistClassifier
from classifiers.ffnn import FeedForwardNeuralNetworkClassifier
from classifiers.cnn import ConvolutionalNeuralNetworkClassifier 

class MnistClassifier:
    def __init__(self, algorithm="rf"):
        if algorithm.lower() == "rf":
            self.model = RandomForestMnistClassifier()
        elif algorithm.lower() == "nn":
            self.model = FeedForwardNeuralNetworkClassifier()
        elif algorithm.lower() == "cnn":
            self.model = ConvolutionalNeuralNetworkClassifier()
        else:
            raise ValueError("Unsupported algorithm. Choose 'rf', 'nn', or 'cnn'.")

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)