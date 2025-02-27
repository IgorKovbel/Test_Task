# MNIST Image Classification with OOP

## Task Description

General requirements for the test:
- The source code should be written in Python 3.
- The code should be clear for understanding and well-commented.
- All solutions should be put into the GitHub repository. Each task should:
  - Be in a separate folder.
  - Contain its own README file with a solution explanation and details on how to set up the project.
  - Include a `requirements.txt` file listing all libraries used in the solution.
- All documentation, comments, and other text information around the project should be written in English.
- The demo should be represented as a Jupyter Notebook and include examples of how your solution works, along with a description of the edge cases.

**Task 1. Image Classification + OOP**

In this task, you need to use a publicly available simple MNIST dataset and build 3 classification models around it. The models should be:
1. **Random Forest**
2. **Feed-Forward Neural Network**
3. **Convolutional Neural Network**

Each model should be implemented as a separate class that adheres to the `MnistClassifierInterface` interface, which declares 2 abstract methods: `train` and `predict`. Finally, all three models are encapsulated under another class named `MnistClassifier`. This class takes an algorithm as an input parameter (possible values: `cnn`, `rf`, and `nn`) and provides predictions with exactly the same structure (inputs and outputs), regardless of the selected algorithm.

The solution includes:
- An interface for models called `MnistClassifierInterface`.
- Three classes (one for each model) that implement `MnistClassifierInterface`.
- A wrapper class, `MnistClassifier`, which takes as an input parameter the name of the algorithm and provides consistent predictions regardless of the selected algorithm.


## Navigate


- **`classifiers/`** – Contains all the classifier implementations and the unified classifier wrapper:
  - **`base.py`** – Defines the abstract interface (`MnistClassifierInterface`) for MNIST classifiers.
  - **`cnn.py`** – Convolutional Neural Network classifier implementation.
  - **`ffn.py`** – Feed-Forward Neural Network classifier implementation.
  - **`random_forest.py`** – Random Forest classifier implementation.
  - **`inference.py`** – Script demonstrating how to load a trained model and perform inference on new data.
  - **`mnist_classifier.py`** – A wrapper class that selects and runs one of the above models based on the specified algorithm.
- **`readme.md`** – Documentation for the project, detailing setup steps and usage instructions.
- **`requirements.txt`** – List of all libraries and dependencies needed to run this solution.
- **`workflow.ipynb`** – Jupyter Notebook demonstrating data loading, training, evaluation, and edge-case handling for all three models.


## Usage


To get results using one of the classifiers (possible values: `cnn`, `rf`, and `nn`), execute the following command:

```sh
python inference.py --algorithm "<algorithm>"
```

## My talk

It was quite an interesting project in which I practiced both OPP and the use of basic machine learning tools.
