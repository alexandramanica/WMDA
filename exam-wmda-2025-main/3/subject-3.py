# Exercițiul 3: Folosiți setul de date despre prețuri imobiliare din California pentru a
# antrena un model de regresie Ridge. Tratați valorile lipsă, folosiți GridSearchCV
# pentru a căuta parametrul optim alpha și afișați eroarea MSE pe setul de test.

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

# target
y = df['median_house_value']

# coloana de predictie este eliminata, coloanele text sunt numerice
X = df.drop('median_house_value', axis=1)
x = X.drop('ocean_proximity', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])

param_grid = {
    "ridge__alpha" : [0.1, 0.01, 1, 10, 100, 1000]}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error"
)

grid_search.fit(x_train, y_train)

best_alpha = grid_search.best_params_["ridge__alpha"]

y_pred = grid_search.predict(x_test)

mse = mean_squared_error(y_test, y_pred)

print(mse)