### Cerinta

# Folositi setul de date despre preturi imobiliare din California pentru a
# antrena un model de regresie `Ridge`.
#
# 1. Eliminati coloana categorica `ocean_proximity` pentru aceasta varianta.
# 2. Separati target-ul `median_house_value` de caracteristicile numerice.
# 3. Tratati valorile lipsa folosind `SimpleImputer(strategy="median")`.
# 4. Standardizati datele cu `StandardScaler`.
# 5. Folositi `GridSearchCV` pentru a cauta:
#
# ```python
# alpha = [0.01, 0.1, 1, 10, 100]
# ```
#
# 6. Afisati cel mai bun `alpha` si eroarea `MSE` pe setul de test.

### Cod De Pornire

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

# === START ===
df = df.drop("ocean_proximity", axis=1)

x = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

param_grid = {
    "ridge__alpha" : [0.01, 0.1, 1, 10, 100]
}

pipeline = Pipeline(
    [
        ("inputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ]
)

grid_search_model = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring = "neg_mean_squared_error" #aici se da ce se cere in cerinta, r2 in general pt regresii sau ce se cere in cerinta, in cazul de fata mse
)

grid_search_model.fit(x_train, y_train)

y_pred = grid_search_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)

print("Mse", mse)
print("Best alpha", grid_search_model.best_params_)
# === END ===