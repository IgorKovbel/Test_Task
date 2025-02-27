from classifiers.base import *


class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),                # Flatten the image
            nn.Linear(28 * 28, 128),     # First fully-connected layer
            nn.ReLU(),
            nn.Linear(128, 64),          # Second fully-connected layer
            nn.ReLU(),
            nn.Linear(64, 10)            # Output layer for 10 classes
        )
        
    def forward(self, x):
        return self.network(x)


class FeedForwardNeuralNetworkClassifier(MnistClassifierInterface):
    def __init__(self, epochs=5, batch_size=64, learning_rate=0.001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the FFNN model, loss function, and optimizer
        self.model = FeedForwardNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, X_train, y_train):
        # Convert training data to torch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        # Create a DataLoader for batch processing
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()