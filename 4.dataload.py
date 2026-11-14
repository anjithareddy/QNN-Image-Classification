from torch.utils.data import TensorDataset, DataLoader

# --- 1. Define Hyperparameters ---
# Since quantum simulators are computationally heavy, we will use a small batch size 
# and a few epochs just to prove the model learns.
batch_size = 16
epochs = 5
learning_rate = 0.01

# --- 2. Create PyTorch DataLoaders ---
# Combine the features and labels into standard PyTorch TensorDatasets
train_dataset_final = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset_final = TensorDataset(X_test_tensor, y_test_tensor)

# DataLoaders handle shuffling and batching the data automatically
train_loader = DataLoader(train_dataset_final, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset_final, batch_size=batch_size, shuffle=False)

print(f"DataLoaders ready! Training batches: {len(train_loader)}")