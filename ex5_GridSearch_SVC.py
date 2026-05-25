
# Folositi setul `Breast Cancer` din `scikit-learn` pentru a clasifica
# observatiile ca maligne sau benigne.
#
# 1. Folositi un `Pipeline` format din `StandardScaler` si `SVC`.
# 2. Cautati parametrii:
#
# ```python
# C = [0.1, 1, 10]
# kernel = ["linear", "rbf"]
# ```
#
# 3. Folositi `GridSearchCV` cu `cv=5`.
# 4. Afisati cei mai buni parametri si accuracy pe test.

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, test_size=0.2)

param_grid = [
        {
            "svc__kernel": ["linear"],
            "svc__C": [0.1, 1, 10],
        },
        {
            "svc__kernel": ["rbf"],
            "svc__C": [0.1, 1, 10],
            "svc__gamma": ["scale", 0.01, 0.1],
        },
    ]

pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("svc", SVC())
    ]
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid= param_grid,
    cv=5,
    n_jobs=-1,
    scoring="accuracy"
)

grid_search.fit(x_train, y_train)

best_param = grid_search.best_params_

y_pred = grid_search.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print(best_param)
print(accuracy)

