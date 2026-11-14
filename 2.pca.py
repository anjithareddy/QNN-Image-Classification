from sklearn.decomposition import PCA

print("Flattening and applying PCA...")

# 1. Flatten the 28x28 images to 1D arrays of length 784
X_train = train_dataset_binary.data.view(-1, 784).numpy() 
y_train = train_dataset_binary.targets.numpy()

X_test = test_dataset_binary.data.view(-1, 784).numpy()
y_test = test_dataset_binary.targets.numpy()

# 2. Initialize PCA to reduce 784 features down to just 4 features
n_features = 4
pca = PCA(n_components=n_features)

# Fit PCA on the training data and transform both train and test sets
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 3. Feature Scaling for Quantum Encoding
# Quantum rotation gates (like RX, RY, RZ) expect angles. 
# We must scale our PCA outputs to be between 0 and pi (π) radians.
def normalize_to_angles(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    # Scales data to [0, 1] then multiplies by π
    return (data - min_val) / (max_val - min_val) * np.pi 

X_train_angles = normalize_to_angles(X_train_pca)
X_test_angles = normalize_to_angles(X_test_pca)

# 4. Convert back to PyTorch Tensors for our Neural Network
X_train_tensor = torch.tensor(X_train_angles, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_angles, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

print(f"Classical Data successfully reduced from 784 to {n_features} features!")
print(f"Shape of X_train_tensor: {X_train_tensor.shape}")