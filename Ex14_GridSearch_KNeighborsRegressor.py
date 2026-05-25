# ## Exercitiul 13: `KNeighborsRegressor` Si Scalarea
#
# **Nivel:** mediu-avansat
# **Concepte:** KNN, distante, `StandardScaler`, alegerea lui `n_neighbors`
#
# ### Cerinta
#
# Creati un model care prezice pretul unei case pe baza suprafetei, numarului
# de dormitoare si locatiei.
#
# 1. Transformati `location` prin `get_dummies()`.
# 2. Impartiti datele in train si test.
# 3. Standardizati caracteristicile.
# 4. Folositi `GridSearchCV` pentru:
#
# ```python
# n_neighbors = [2, 3, 4, 5]
# weights = ["uniform", "distance"]
# ```
#
# 5. Afisati parametrii alesi si `MAE` pe test.

### Cod De Pornire

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

data = {
    "location": ["cityA", "cityB", "cityA", "cityB", "cityA", "cityB",
                 "cityA", "cityB", "cityA", "cityB", "cityA", "cityB"],
    "sqft": [900, 1000, 1100, 1250, 1350, 1450, 1550, 1700, 1850, 2000, 2150, 2300],
    "bedrooms": [2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5],
    "price": [175000, 220000, 215000, 265000, 250000, 295000,
              300000, 345000, 350000, 395000, 405000, 455000]
}

df = pd.DataFrame(data)

df = pd.get_dummies(df, columns=["location"], drop_first=True)

x = df.drop(columns="price", axis=1)
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pipeline = Pipeline(
    [("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor())]
)

# prefixul de la param e prefixul din model (ex knn)
param_grid = {
    "knn__n_neighbors": [2, 3, 4, 5],
    "knn__weights": ["uniform", "distance"]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring="neg_mean_absolute_error"
)

grid_search.fit(x_train, y_train)
y_pred = grid_search.predict(x_test)

best_coef = grid_search.best_params_
mae = mean_absolute_error(y_test, y_pred)

print("Best coef ", best_coef)
print("MAE", mae)