### Cerinta

# Generati date pentru detectarea fraudei, cu aproximativ 98% tranzactii
# legitime si 2% fraude.
#
# 1. Folositi `LogisticRegression`.
# 2. Cautati prin `GridSearchCV` combinatiile:
#
# ```python
# C = [0.01, 0.1, 1, 10]
# class_weight = [None, "balanced"]
# ```
#
# 3. Alegeti modelul folosind `scoring="recall"`.
# 4. Afisati:
#
# ```text
# best_params_
# recall pe test
# classification_report
# ```

### Cod De Pornire

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score, classification_report

x, y = make_classification(
    n_samples=3000,
    n_features=8,
    n_informative=4,
    weights=[0.98, 0.02],
    random_state=42
)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "class_weight": [None, "balanced"]
}

#Exempluu
# param_grid = {
#     "model__C": [0.01, 0.1, 1, 10],
#     "model__class_weight": [None, "balanced"]
# }

grid_search = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring="recall"
)

grid_search.fit(x_train, y_train)
y_pred = grid_search.predict(x_test)

best_params = grid_search.best_params_
rec_score = recall_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(rec_score)
print(report)
print(best_params)