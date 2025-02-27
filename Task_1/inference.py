import time
import argparse
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mnist_classifier import MnistClassifier

def load_data():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.astype(np.float32) / 255.0  # Normalize pixel values
    y = mnist.target.astype(np.int64)
    return X, y

def main(args):
    X, y = load_data()
    
    classifier = MnistClassifier(algorithm=args.algorithm)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    start_time = time.time()

    classifier.train(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Random Forest Accuracy:", accuracy)
    print("Time taken: {:.2f} seconds\n".format(time.time() - start_time))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MNIST Inference Script")
    parser.add_argument('--algorithm', type=str, default='rf', choices=['rf', 'nn', 'cnn'],
                        help="Select algorithm: 'rf' for Random Forest, 'nn' for Feed-Forward NN, 'cnn' for Convolutional NN")
    args = parser.parse_args()
    main(args)
