from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import pandas as pd

import flwr as fl


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy            


def fit_evaluate(parameters, config):
    """
    This function handles both training and evaluation of the linear regression model
    on a client in the federated learning setting using Flower.

    Args:
        parameters: A list containing the model coefficients and intercept received
                    from the Flower server.
        config: The configuration object from Flower, potentially containing information
                about the current round or other settings.

    Returns:
        A tuple containing:
            - Updated model parameters (coefficients and intercept) as a list.
            - Number of training samples on the client as an integer.
            - A dictionary containing evaluation metrics (e.g., MSE, R-squared).
    """

    # 1. Load Data (Replace with your client-specific data loading logic)
    # This part needs to be adapted to how you access and prepare data on each client.
    # Here's an example assuming CSV files:

    client_id = config["client_id"]  # Access client ID from config
    data_path = f"C:\\Users\\eshiv\\train_{client_id}.csv"  # Replace with your path
    data_path_test = f"C:\\Users\\eshiv\\test_{client_id}.csv"  # Replace with your path

    

    try:
        train_df = pd.read_csv(data_path)
        test_df = pd.read_csv(data_path_test)
    except FileNotFoundError:
        print(f"Client {client_id}: Train data not found at {data_path}")
        return parameters, 0, {}  # Return empty data if file not found

    # 5. Evaluate the Model on the Local Data (optional)
    replacement_dict = {'162+': 162}

    # Replace values in PayDelay column using the dictionary
    train_df['PayDelay'] = train_df['PayDelay'].replace(replacement_dict)
    test_df['PayDelay'] = test_df['PayDelay'].replace(replacement_dict)

    selected_features = [
        'PayDelay',
        # ... (other features if needed)
    ]

    X_train = train_df[selected_features]
    y_train = train_df['DaysInHospital']
    X_test = test_df[selected_features]
    y_test = test_df['DaysInHospital'] 

    # 2. Extract Model Parameters from FL
    model_coef, model_intercept = parameters

    # 3. Create a LinearRegression Model
    model = LinearRegression(coef=model_coef, intercept=model_intercept)

    # 4. Train the Model on the Local Data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    # 6. Calculate Metrics (replace with your desired metrics)
    mse = mean_squared_error(y_test, y_pred)  # Assuming you have y_test for local evaluation

    # 7. Return Updated Parameters and Evaluation Metrics
    return [model.coef_, model.intercept_], len(X_train), {"mse": mse}


def load_model():
    model = Net().to(DEVICE)
    return model

if __name__ == "__main__":
    net =  load_model()
    trainloader, testloader, num_examples = load_data()
    train(net,trainloader,1)
    loss , accuracy = test(net,testloader)
    print("loss: ,{loss:.5f}, accuracy: ,{accuracy:.5f}")

