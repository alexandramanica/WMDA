# Folositi setul de date `Wine` pentru a prezice clasa vinului.
#
# 1. Impartiti datele in train si test.
# 2. Folositi `GridSearchCV` pentru a cauta cea mai buna valoare pentru:
# 3. Folositi `cv=5` si `scoring="accuracy"`.
# 4. Afisati cei mai buni parametri, scorul mediu de validare si accuracy pe test.

### Cod De Pornire

from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

wine = load_wine()

x = wine.data
y = wine.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2 ,random_state=42)

param_grid = {
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

best_param = grid_search.best_params_
best_score = grid_search.best_score_
acc_score = accuracy_score(y_test, y_pred)

print(best_param)
print(best_score)
print(acc_score)