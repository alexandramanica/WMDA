# Exercițiul: Folosiți setul de date Iris din scikit-learn.
# Clasificați specia florii (target) pe baza celor 4 caracteristici.
#
# 1. Împărțiți datele în train și test (test_size=0.2)
# 2. Antrenați un KNeighborsClassifier(n_neighbors=5)
#    (nu uita scalarea unde e cazul)
# 3. Testați cu n_neighbors = [3, 5, 7, 9] și afișați accuracy pentru fiecare
# 4. Afișați classification_report pentru cel mai bun n_neighbors

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score, classification_report

data = load_iris()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

n_neighbors = [3, 5, 7, 9]

best_accuracy = 0
best_y_pred = None
best_n = None

for n in n_neighbors:
    model_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kn_neighbours", KNeighborsClassifier(n_neighbors=n))
        ]
    )

    model_pipeline.fit(x_train, y_train)
    y_pred = model_pipeline.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n = n
        best_y_pred = y_pred


print("Neighbours = ", best_n)
print("Accuracy", accuracy_score(y_test, best_y_pred))
print("Classreport", classification_report(y_test, best_y_pred))
