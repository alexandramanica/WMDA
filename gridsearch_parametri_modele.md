# GridSearchCV – Parametri uzuali pentru modele

`GridSearchCV` se folosește pentru a testa automat mai multe combinații de hiperparametri și pentru a alege varianta cea mai bună pe baza unei metrici.

---

## 1. Modele de regresie

### Ridge

Parametru important: `alpha`

```python
param_grid = {
    "alpha": [0.01, 0.1, 1, 10, 100],
    "solver": ["auto", "svd", "cholesky", "lsqr"]
}
```

---

### Lasso

Parametru important: `alpha`

```python
param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1, 10],
    "max_iter": [1000, 5000, 10000]
}
```

---

### ElasticNet

Parametri importanți: `alpha`, `l1_ratio`

```python
param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1],
    "l1_ratio": [0.1, 0.5, 0.9],
    "max_iter": [1000, 5000]
}
```

---

### DecisionTreeRegressor

Parametri importanți: `max_depth`, `min_samples_split`, `min_samples_leaf`

```python
param_grid = {
    "max_depth": [None, 3, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "criterion": ["squared_error", "absolute_error"]
}
```

---

### RandomForestRegressor

Parametri importanți: `n_estimators`, `max_depth`

```python
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
```

---

### SVR

Parametri importanți: `C`, `kernel`, `gamma`, `epsilon`

```python
param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf", "poly"],
    "gamma": ["scale", "auto"],
    "epsilon": [0.01, 0.1, 1]
}
```

---

### KNeighborsRegressor

Parametri importanți: `n_neighbors`, `weights`, `metric`

```python
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}
```

---

## 2. Modele de clasificare

### LogisticRegression

Parametri importanți: `C`, `penalty`, `solver`

Pentru `l2`:

```python
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l2"],
    "solver": ["lbfgs", "liblinear"],
    "max_iter": [1000]
}
```

Pentru `l1`:

```python
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l1"],
    "solver": ["liblinear", "saga"],
    "max_iter": [1000]
}
```

---

### DecisionTreeClassifier

Parametri importanți: `max_depth`, `min_samples_split`, `min_samples_leaf`

```python
param_grid = {
    "max_depth": [None, 3, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "criterion": ["gini", "entropy"]
}
```

---

### RandomForestClassifier

Parametri importanți: `n_estimators`, `max_depth`

```python
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}
```

---

### SVC

Parametri importanți: `C`, `kernel`, `gamma`

```python
param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf", "poly"],
    "gamma": ["scale", "auto"]
}
```

---

### KNeighborsClassifier

Parametri importanți: `n_neighbors`, `weights`, `metric`

```python
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}
```

---

### GradientBoostingClassifier

Parametri importanți: `n_estimators`, `learning_rate`, `max_depth`

```python
param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [2, 3, 5]
}
```

---

## 3. Regresie polinomială cu Pipeline

Aici parametrul cel mai important este `degree` din `PolynomialFeatures`.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

pipeline = Pipeline([
    ("poly", PolynomialFeatures()),
    ("model", LinearRegression())
])

param_grid = {
    "poly__degree": [1, 2, 3, 4, 5, 6, 7, 8, 9]
}
```

Forma generală pentru parametrii dintr-un pipeline este:

```text
nume_pas_pipeline__nume_parametru
```

Exemplu:

```python
"poly__degree"
```

înseamnă parametrul `degree` din pasul numit `poly`.

---

## 4. Exemplu complet GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="r2"
)

grid.fit(x_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
```

Corect:

```python
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="r2"
)
```

---

## 5. Rezumat pentru examen/laborator

| Model | Parametri importanți |
|---|---|
| PolynomialFeatures | `degree` |
| Ridge | `alpha` |
| Lasso | `alpha` |
| ElasticNet | `alpha`, `l1_ratio` |
| DecisionTree | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| RandomForest | `n_estimators`, `max_depth` |
| KNN | `n_neighbors`, `weights`, `metric` |
| SVM / SVC / SVR | `C`, `kernel`, `gamma` |
| LogisticRegression | `C`, `penalty`, `solver` |
| GradientBoosting | `n_estimators`, `learning_rate`, `max_depth` |

---

## 6. Observație importantă

`GridSearchCV` nu este legat de un singur model. Poate fi folosit cu aproape orice estimator din `scikit-learn`, atâta timp cât modelul are hiperparametri care pot fi testați.

