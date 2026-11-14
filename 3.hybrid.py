import pennylane as qml
import torch.nn as nn

# --- 1. Define the Quantum Circuit (QNode) ---
n_qubits = n_features # 4 qubits to match our 4 PCA features
dev = qml.device("default.qubit", wires=n_qubits)

# Depth is the number of trainable quantum layers. 
# 2 is perfect for proving the concept on a laptop without taking hours to train.
q_depth = 2 

@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    # Step A: Data Encoding (Classical -> Quantum)
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # Step B: Trainable Quantum Layers
    # This applies a sequence of rotations and CNOT gates based on 'weights'
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Step C: Measurement (Quantum -> Classical)
    # We measure the expectation value of each qubit along the Z-axis
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


# --- 2. Wrap it into a PyTorch Hybrid Model ---
# We must tell PyTorch the shape of the weights the quantum circuit expects
weight_shapes = {"weights": (q_depth, n_qubits, 3)}

class HybridQNN(nn.Module):
    def __init__(self):
        super(HybridQNN, self).__init__()
        
        # 1. The Quantum Layer (PennyLane handles the complex math behind the scenes)
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        
        # 2. The Classical Output Layer
        # Maps the 4 output values from the qubits to 2 probabilities (T-shirt vs Trouser)
        self.clayer = nn.Linear(n_qubits, 2)
        
    def forward(self, x):
        # Forward pass: Data goes through Quantum, then Classical
        x = self.qlayer(x)
        x = self.clayer(x)
        return x

# Initialize the model
model = HybridQNN()
print("Hybrid Quantum-Classical Model successfully initialized!")