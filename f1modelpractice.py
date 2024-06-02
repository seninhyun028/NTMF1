'''
NTM Model for F1 Data
'''

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Step 1: Data Processing
data = pd.read_excel('DATA.xlsx')

# Step 2: Controller
class Controller(nn.Module):
    ''' Controller'''
    def __init__(self, input_size, hidden_size):
        super(Controller,self).__init__()
        self.input_size = input_size
        self.hidden_state = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self,x):
        ''' Forward Pass'''
        # Implement the forward pass of NTM Model
        output, (h_n, c_n) = self.rnn(x)
        return output, (h_n, c_n)

# Step 3: Memory
class NTMMemory(nn.Module) :
    '''Memory'''
    def __init__(self, memory_size, memory_dim):
        super(NTMMemory, self).__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.memory = nn.Parameter(torch.randn(memory_size,memory_dim), requires_grad=False)
    def forward (self, *args, **kwargs):
        '''Forward pass'''
    def read (self, read_weights):
        '''Read weights'''
        return torch.matmul(read_weights.unsqueeze(1),self.memory). squeeze(1)
    def write (self, write_weights, add_vector):
        ''' Write weights'''
        # Retained Information
        retained_memory = (1 - write_weights.unsqueeze(1)) * self.memory

        # New Information
        new_information = write_weights.unsqueeze(-1) * add_vector.unsqueeze(1)

        # Update Memory
        self.memory = retained_memory + new_information

    def content_adressing(self, keys, strength):
        '''Content adressing'''
        # Perform content based adressing
        normalized_memory = F.normalize(self.memory, p=2, dim=1)
        normalized_keys = F.normalize(keys, p=2, dim=-1)
        similarity = torch.matmul(normalized_memory, normalized_keys.unsqueeze(-1).squeeze(1))
        return F.softmax(similarity * strength, dim=-1)

# Step 4: NTM Model (Controller + Memory)
class NTM(nn.Module):
    '''NTM Model'''
    def __init__(self, input_size, output_size, hidden_size, memory_size, memory_dim):
        super(NTM, self).__init__()
        self.controller = Controller(input_size, hidden_size)
        self.memory = NTMMemory(memory_size,memory_dim)
        self.fc = nn.Linear(memory_dim, output_size)
    def forward(self,x):
        '''Forward pass'''
        output, (h_n, _) = self.controller(x)
        read_weights = self.memory.content_adressing(h_n, strength=1)
        read_data = self.memory.read (read_weights)
        output = self.fc(read_data)
        return output

# Step 5: Training

def prepare_data(input_data, target_column='Optimal Performance'):
    ''' Prepare data '''
    # 1) Extract features and labels from data
    features = input_data.drop(columns=[target_column])
    labels = input_data[target_column]

    # 2) Specify the categorical columns to be encoded
    categorical_columns = ['Grand Prix', 'Location', 'Circuit', 'Type',
                           'Tyre Compounds', '1st Pit Stop', '2nd Pit Stop',
                           '3rd Pit Stop', '4th Pit Stop', '5th Pit Stop', '6th Pit Stop']

    # 3) One-Hot Encode categorical columns
    encoder = OneHotEncoder(categories='auto')
    encoded_cats = encoder.fit_transform(features[categorical_columns])
    encoded_cats_df = pd.DataFrame(encoded_cats.toarray(), columns=encoder.
                                   get_feature_names_out(categorical_columns))

    # 4) Drop the original categorical columns and reset the index
    features = features.drop(columns=categorical_columns).reset_index(drop=True)

    # 5) Combine the encoded columns with the remaining numeric data
    features = pd.concat([features, encoded_cats_df], axis=1)

    # 6) Ensure all data is numeric
    features = features.apply(pd.to_numeric)

    # 7) Convert features and labels to tensors
    x_tensor = torch.tensor(features.values, dtype=torch.float32)
    y_tensor = torch.tensor(labels.values, dtype=torch.float32)

    # 8) Create TensorDataset
    return TensorDataset(x_tensor, y_tensor)

def build_model(input_size, output_size, hidden_state, memory_size, memory_dim):
    ''' Build model'''
    # 1) Instatntiate NTM Model
    model_to_train = NTM(input_size, output_size, hidden_state, memory_size, memory_dim)
    return model_to_train

# Prepare Data
dataset = prepare_data(data)

# Dertermine input size
INPUT_SIZE = dataset.tensors[0].shape[1]

# Model Parameters (subject to change)
OUTPUT_SIZE = 1
HIDDEN_SIZE = 64
MEMORY_SIZE = 128
MEMORY_DIM = 20

# Build Model
model = build_model(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, MEMORY_SIZE, MEMORY_DIM)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
BATCH_SIZE = 32
EPOCHS = 10

train_loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

def train_model(epochs):
    ''' Train model'''
    # Step 1) Iterate over the specified number of epochs
    for epoch in range(epochs):
        model.train() # Set the model to training model
        running_loss = 0.0 # Initiliaze running loss for the epoch
        # Step 2) Iterate over the batches in the training DataLoader
        for inputs, labels in train_loader:
            optimizer.zero_grad() # Zero the gradients to avoid accumalation (CHECK)
            outputs = model(inputs) # Forward Pass: compute model predictions (CHECK)
            loss = criterion(outputs, labels) # Loss Function ( Prediction vs. Actual Labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0) # Update running loss for the epoch (CHECK)
        epoch_loss = running_loss / len(train_loader.dataset) # Calculate avg. loss for the epoch
        print(f'Epoch [{epoch + 1} / {epochs}], Loss: {epoch_loss: .4f}')

train_model(EPOCHS) # Print epoch number and avg.loss

# Step 6: Iteration
def get_grand_prix_details(trained_model, grand_prix_name, grand_prix_data,
                           grand_prix_column='Grand Prix',
                           score_column='Optimal Performance Score'):
    ''' Grand Prix Details'''
# 1) Filter data for the specified Grand Prix name
    grand_prix_data_filtered = grand_prix_data[grand_prix_data
                                               [grand_prix_column] == [grand_prix_name]]
    if grand_prix_data.empty:
        raise ValueError(f"Grand Prix '{grand_prix_name}'not found in the dataset")
    features = grand_prix_data_filtered.drop(columns=['Grand Prix'])
# 2) Remove Grand Prix column (not a feature)
    x = torch.tensor(features.values, dtype=torch.float32)
# 3) Extract the target column
    y = torch.tensor(grand_prix_data_filtered[score_column].values, dtype=torch.float32)
# 3) Perform inference using trained model
    with torch.no_grad():
        trained_model.eval()
        outputs = model(x)
# 4) Post-process model outputs if needed
    scaled_predictions = scale_predictions(outputs)
 # 5) Create dictionary to store details for the specific Grand Prix
    grand_prix_details = {
        'Grand Prix Name' : grand_prix_name,
        'Optimal Performance Score': scaled_predictions, 
        'Target Score Column': y.tolist(),
        # Add other details based on model outputs or post-processing
    }
    return grand_prix_details

def scale_predictions(predictions):
    ''' Scale predictions''' 
    # 6) Scale predictions to a specific range (0 to 100)
    scaled_predictions = predictions * 100 # Assuming predictions are normalized between 0 and 1
    return scaled_predictions

# Step 7: Evaluation

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_dataset = prepare_data(train_data)
test_dataset = prepare_data(test_data)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_model(epochs=EPOCHS)

def evaluate_model(eval_model, data_loader):
    ''' Evaluate model'''
    eval_model.eval() # Set model to evaluation model
    predictions, actuals = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            predictions.append(outputs.numpy())
            actuals.append(labels.numpy())
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    mse_val = mean_squared_error(actuals, predictions)
    mae_val = mean_absolute_error(actuals, predictions)
    r2_val = r2_score(actuals, predictions)

    return mse_val, mae_val, r2_val

mse, mae, r2 = evaluate_model(model, test_loader)
print (f'Test MSE: {mse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}')
