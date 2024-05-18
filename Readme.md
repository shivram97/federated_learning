# Federated Learning for Linear Regression

This repository implements a federated learning framework for training a linear regression model using Flower. It enables distributed training on client devices while preserving data privacy.

## Requirements

- Python 3.6+
- Flower (`pip install flwr`)
- scikit-learn (`pip install scikit-learn`)
- PyTorch (optional, for GPU acceleration; `pip install torch`)

## Project Structure

federated_learning/
	client.py # Client script for Flower
	centralized.py # Functions for data loading, model training, and evaluation
	README.md # This file (instructions and usage)
	server.py # Flower server script

perl


## Instructions

### Data Preparation

Create CSV files named `train_{client_id}.csv` and `test_{client_id}.csv` for each client, containing training and testing data in the following format (replace with your actual features and target):

```plaintext
PayDelay, ... (other features)
10, ...
15, ...
...```
Ensure these files are placed in client-specific directories named client_0, client_1, etc., within your project directory.

Configuration (Optional)
You can modify the data paths in centralized.py if your data structure differs.

Run the Server
Open a terminal and navigate to your project directory.

Start the Flower server:

bash

python server.py
Run Client Scripts
Open multiple terminals (one for each simulated client).

In each terminal, navigate to your project directory and run the client script with appropriate arguments:

terminal

python client.py --server_address localhost:8080 --client_id 0  # Client 1
python client.py --server_address localhost:8080 --client_id 1  # Client 


# ... (more clients if needed)
Replace 0, 1, etc. with unique client IDs.

Understanding the Code
client.py
Establishes a connection to the Flower server.
Implements the FlowerClient class to handle model updates, training, and evaluation on clients.
centralized.py
Provides functions for:

Data loading from client-specific CSV files (load_data).
Model loading (creates a PyTorch linear regression model; load_model).
Training and evaluation on a client using received model parameters (fit_evaluate).
Performs data preprocessing (replacements for categorical data) and calculates evaluation metrics (MSE, R-squared).
server.py
Configures the Flower server (number of rounds, strategy).
Starts the server, listening for client connections.



#Note: This project only shows the simulation in which a federated model runs. -(the parameters are not optimized)
#Run each dataset through the the preprocessing steps and then run the federated model using the above steps.