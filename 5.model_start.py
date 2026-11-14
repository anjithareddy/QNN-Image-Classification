import torch.optim as optim

# --- 1. Define Loss and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to keep track of our progress for plotting later
train_losses = []
train_accuracies = []

print("Starting Quantum Model Training...")

# --- 2. The Training Loop ---
for epoch in range(epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Iterate through the data in batches
    for batch_features, batch_labels in train_loader:
        
        # Step A: Zero the gradients (clear old calculations)
        optimizer.zero_grad()
        
        # Step B: Forward Pass (Push data through the quantum circuit)
        outputs = model(batch_features)
        
        # Step C: Calculate the Loss (How wrong the model was)
        loss = criterion(outputs, batch_labels)
        
        # Step D: Backward Pass (Calculate gradients using PennyLane's automatic differentiation)
        loss.backward()
        
        # Step E: Optimize (Update the classical weights and quantum rotation angles)
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        
        # Calculate accuracy for this batch
        _, predicted = torch.max(outputs.data, 1)
        total_samples += batch_labels.size(0)
        correct_predictions += (predicted == batch_labels).sum().item()
        
    # Epoch summaries
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = (correct_predictions / total_samples) * 100
    
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

print("Training Complete!")