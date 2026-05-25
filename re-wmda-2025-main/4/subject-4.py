# Exercițiul 4: Antrenați o rețea neuronală simplă (cu un strat ascuns) în PyTorch
# pentru clasificarea cifrelor folosind setul de date load_digits din sklearn.
# Afișați acuratețea pe setul de test după 10 de epoci.

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === Your code starts here ===

torch.manual_seed(42)

# 1. Incarcarea datelor
digits = load_digits()

x = digits.data       # fiecare imagine 8x8 este transformata in 64 de valori
y = digits.target     # cifra corecta: 0, 1, ..., 9

# 2. Impartirea datelor in train si test
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Standardizarea caracteristicilor
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 4. Transformarea datelor in tensori PyTorch
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 5. Definirea retelei neuronale cu un singur strat ascuns
class DigitsNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_layer = nn.Linear(64, 32)  # 64 pixeli -> 32 neuroni ascunsi
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(32, 10)  # 32 neuroni -> 10 cifre

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)

        return x


model = DigitsNeuralNetwork()

# 6. Functia de pierdere si optimizatorul
# Avem 10 clase, deci folosim CrossEntropyLoss.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 7. Antrenarea modelului pentru 10 epoci
epochs = 10

for epoch in range(epochs):
    model.train()

    # Predictii pentru datele de antrenare
    outputs = model(x_train_tensor)

    # Eroarea fata de cifrele reale
    loss = criterion(outputs, y_train_tensor)

    # Calcularea gradientilor si actualizarea ponderilor
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 8. Evaluarea pe setul de test
model.eval()

with torch.no_grad():
    test_outputs = model(x_test_tensor)

    # Alegem cifra cu scorul cel mai mare
    _, y_pred = torch.max(test_outputs, dim=1)

    correct_predictions = (y_pred == y_test_tensor).sum().item()
    total_predictions = y_test_tensor.size(0)

    accuracy = correct_predictions / total_predictions

print(f"Test Accuracy: {accuracy:.2f}")

# === Your code ends here ===