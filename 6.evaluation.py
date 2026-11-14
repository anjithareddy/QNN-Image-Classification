import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch

# --- 1. Evaluate on Test Data ---
print("Evaluating model on test dataset...")
model.eval() # Set model to evaluation mode so weights do not change

all_preds = []
all_labels = []

with torch.no_grad(): # Turn off gradient tracking for faster testing
    for batch_features, batch_labels in test_loader:
        outputs = model(batch_features)
        
        # Get the highest probability class
        _, predicted = torch.max(outputs, 1)
        
        # Store predictions and true labels for metric calculations
        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_labels.numpy())

# --- 2. Print Classification Report ---
# This automatically calculates Precision, Recall, and F1-Score
print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=['T-shirt/top', 'Trouser']))

# --- 3. Generate Professional Visualizations ---
# We will create a 1x3 grid of charts for your project report
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Chart A: Training Loss
axes[0].plot(train_losses, label='Training Loss', color='#e74c3c', marker='o')
axes[0].set_title('Training Loss over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].legend()

# Chart B: Training Accuracy
axes[1].plot(train_accuracies, label='Training Accuracy', color='#3498db', marker='o')
axes[1].set_title('Training Accuracy over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].legend()

# Chart C: Confusion Matrix Heatmap
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2], 
            xticklabels=['T-shirt', 'Trouser'], yticklabels=['T-shirt', 'Trouser'])
axes[2].set_title('Test Data Confusion Matrix')
axes[2].set_xlabel('Predicted Label')
axes[2].set_ylabel('True Label')

plt.tight_layout()
plt.show()