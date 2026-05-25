# Exercițiul 4: Implementați o rețea neuronală simplă folosind PyTorch pentru a clasifica
# datele din setul Iris. Antrenați modelul pentru 100 de epoci și afișați acuratețea pe
# setul de test.

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === START ===

# Pentru a obtine aceleasi rezultate la fiecare rulare
torch.manual_seed(42)

# 1. Incarcarea datelor
iris = load_iris()

x = iris.data       # 4 caracteristici: lungime/latime sepala si petala
y = iris.target     # 3 clase de flori: 0, 1, 2

# 2. Impartirea datelor in train si test
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Standardizarea caracteristicilor
# Retelele neuronale invata mai bine cand valorile sunt pe scari apropiate.
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 4. Transformarea datelor in tensori PyTorch
# Pentru intrari folosim float32.
# Pentru clase folosim long, deoarece CrossEntropyLoss asteapta etichete intregi.
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 5. Definirea retelei neuronale
class IrisNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(4, 10)   # 4 feature-uri de intrare -> 10 neuroni
        self.relu = nn.ReLU()            # functie de activare
        self.layer2 = nn.Linear(10, 3)   # 10 neuroni -> 3 clase

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x


model = IrisNeuralNetwork()

# 6. Functia de pierdere si optimizatorul
# CrossEntropyLoss se foloseste pentru clasificare multiclas.
criterion = nn.CrossEntropyLoss()

# Adam actualizeaza ponderile retelei in timpul invatarii.
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 7. Antrenarea modelului pentru 100 de epoci
epochs = 100

for epoch in range(epochs):
    # Punem modelul in modul de antrenare
    model.train()

    # Calculam predictiile pe datele de train
    outputs = model(x_train_tensor)

    # Calculam eroarea dintre predictii si clasele reale
    loss = criterion(outputs, y_train_tensor)

    # Stergem gradientii calculati la pasul anterior
    optimizer.zero_grad()

    # Calculam gradientii actuali
    loss.backward()

    # Actualizam ponderile modelului
    optimizer.step()

    # Afisam pierderea la fiecare 10 epoci
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 8. Evaluarea modelului pe setul de test
model.eval()

# La evaluare nu este nevoie sa calculam gradienti.
with torch.no_grad():
    test_outputs = model(x_test_tensor)

    # Pentru fiecare floare alegem clasa cu scorul cel mai mare.
    _, y_pred = torch.max(test_outputs, dim=1)

    # Calculam acuratetea.
    correct_predictions = (y_pred == y_test_tensor).sum().item()
    total_predictions = y_test_tensor.size(0)

    accuracy = correct_predictions / total_predictions

print(f"Test Accuracy: {accuracy:.2f}")

# === END ===
