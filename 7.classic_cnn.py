import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# --- 1. Define the Classical CNN ---
# We will use a lightweight CNN for a fair comparison against our small QNN
class ClassicalCNN(nn.Module):
    def __init__(self):
        super(ClassicalCNN, self).__init__()
        # 1 input channel (grayscale), 4 output channels (filters), 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pooling, the 28x28 image becomes 14x14. 
        # Flattened size: 4 channels * 14 * 14 = 784 classical features
        self.fc = nn.Linear(784, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten for the linear layer
        x = self.fc(x)
        return x

cnn_model = ClassicalCNN()

# --- 2. Prepare DataLoaders for Original Images ---
# The CNN needs the original 2D images, not the 1D PCA features we used for the QNN
cnn_train_loader = DataLoader(train_dataset_binary, batch_size=16, shuffle=True)
cnn_test_loader = DataLoader(test_dataset_binary, batch_size=16, shuffle=False)

# --- 3. Train the Classical CNN ---
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
cnn_criterion = nn.CrossEntropyLoss()

print("Starting Classical CNN Training...")
start_time = time.time()

cnn_epochs = 5
for epoch in range(cnn_epochs):
    running_loss = 0.0
    for images, labels in cnn_train_loader:
        cnn_optimizer.zero_grad()
        
        # Forward pass
        outputs = cnn_model(images)
        loss = cnn_criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        cnn_optimizer.step()
        
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(cnn_train_loader)
    print(f"Epoch [{epoch+1}/{cnn_epochs}] | Loss: {epoch_loss:.4f}")

cnn_time = time.time() - start_time
print(f"Classical CNN Training Complete in {cnn_time:.2f} seconds!")