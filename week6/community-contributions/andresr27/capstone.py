import os
from dotenv import load_dotenv
from huggingface_hub import login
from pricer.evaluator import evaluate
from litellm import completion
from pricer.items import Item
import numpy as np
from tqdm import tqdm
import csv
from sklearn.feature_extraction.text import HashingVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR

LITE_MODE = True

load_dotenv(override=True)
hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)
username = "ed-donner"
dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"

train, val, test = Item.from_hub(dataset)

print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items")

# Write the test set to a CSV
# Only write if human_in.csv doesn't exist to prevent overwriting manual input
if not os.path.exists('human_in.csv'):
    with open('human_in.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for t in test[:100]:
            writer.writerow([t.summary, 0])
else:
    print("human_in.csv already exists. Skipping writing to prevent overwrite.")

# Read it back in

human_predictions = []
# Check if human_out.csv exists before attempting to read it
if os.path.exists('human_out.csv'):
    with open('human_out.csv', 'r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            human_predictions.append(float(row[1]))
else:
    print("human_out.csv not found. Human predictions will be empty.")

def human_pricer(item):
    idx = test.index(item)
    # Ensure human_predictions is not empty before accessing elements
    if human_predictions:
        return human_predictions[idx]
    else:
        # Return a default value or raise an error if predictions are missing
        print("Warning: human_predictions is empty. Returning 0.")
        return 0

human = human_pricer(test[0])
actual = test[0].price
print(f"Human predicted {human} for an item that actually costs {actual}")

evaluate(human_pricer, test, size=100)

# Prepare our documents and prices

y = np.array([float(item.price) for item in train])
documents = [item.summary for item in train]

# Use the HashingVectorizer for a Bag of Words model
# Using binary=True with the CountVectorizer makes "one-hot vectors"

np.random.seed(42)
vectorizer = HashingVectorizer(n_features=5000, stop_words='english', binary=True)
X = vectorizer.fit_transform(documents)


# Define the neural network - here is Pytorch code to create a 8 layer neural network with Dropout and Batchnorm

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.layer3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, 32)
        self.layer6 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        return self.layer6(x)


# Convert data to PyTorch tensors (modified to avoid X.toarray() on entire dataset)
# First, split the sparse SciPy matrix X and numpy array y
X_train_sparse, X_val_sparse, y_train_split, y_val_split = train_test_split(
    X, y, test_size=0.01, random_state=42
)

# Now convert the split sparse SciPy matrices to dense PyTorch FloatTensors
# This avoids calling .toarray() on the entire X dataset at once, reducing memory footprint.
X_train_tensor = torch.FloatTensor(X_train_sparse.toarray())
X_val_tensor = torch.FloatTensor(X_val_sparse.toarray())
y_train_tensor = torch.FloatTensor(y_train_split).unsqueeze(1)
y_val_tensor = torch.FloatTensor(y_val_split).unsqueeze(1)

# Create the loader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model
input_size = X_train_tensor.shape[1]
model = NeuralNetwork(input_size)
# %%
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {trainable_params:,}")
# %%
# Define loss function and optimizer

loss_function = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# We will do 10 complete runs through the data

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    for batch_X, batch_y in tqdm(train_loader):
        optimizer.zero_grad()

        # The next 4 lines are the 4 stages of training: forward pass, loss calculation, backward pass, optimize

        outputs = model(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = loss_function(val_outputs, y_val_tensor)

    print(f'Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {loss.item():.3f}, Val Loss: {val_loss.item():.3f}')


#
def neural_network(item):
    model.eval()
    with torch.no_grad():
        vector = vectorizer.transform([item.summary])
        # Convert single item sparse vector to dense for model input
        vector = torch.FloatTensor(vector.toarray())
        result = model(vector)[0].item()
    return max(0, result)

evaluate(neural_network, test)
