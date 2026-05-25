# Exercițiul 3: Descărcați setul de date despre automobile de la URL-ul de mai jos.
# Folosiți regresie polinomială (gradul 2) pentru a prezice consumul de combustibil
# pe baza puterii motorului. Afișați R² pe setul de test.

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

url = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/cars.csv"
df = pd.read_csv(url)

# === Your code starts here ===
# Folosiți coloanele 'Horsepower' (predictor) și 'MPG' (target)

data = df[["hp", "mpg"]].dropna()

x = data[["hp"]]
y = data[["mpg"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)

print(r2)

# === Your code ends here ===
