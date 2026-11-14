# --- Evaluate Classical CNN on Test Data ---
print("Evaluating Classical CNN on test dataset...")
cnn_model.eval() # Set model to evaluation mode

cnn_correct = 0
cnn_total = 0
cnn_all_preds = []
cnn_all_labels = []

with torch.no_grad():
    for images, labels in cnn_test_loader:
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        cnn_total += labels.size(0)
        cnn_correct += (predicted == labels).sum().item()
        
        cnn_all_preds.extend(predicted.numpy())
        cnn_all_labels.extend(labels.numpy())

cnn_accuracy = (cnn_correct / cnn_total) * 100
print(f"\nClassical CNN Test Accuracy: {cnn_accuracy:.2f}%")

print("\n--- Classical CNN Classification Report ---")
print(classification_report(cnn_all_labels, cnn_all_preds, target_names=['T-shirt/top', 'Trouser']))