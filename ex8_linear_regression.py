## Exercitiul 8: Publicitate Si Vanzari

### Cerinta

# Folositi bugetele pentru TV, Radio si Newspaper pentru a estima vanzarile.
#
# 1. Separati `X` si `y`.
# 2. Impartiti datele in train si test.
# 3. Antrenati un model `LinearRegression`.
# 4. Afisati coeficientii, interceptul, `R2`, `MSE` si `MAE`.

### Cod De Pornire

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

data = {
    "TV": [230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.8, 66.1, 214.7],
    "Radio": [37.8, 39.3, 45.9, 41.3, 10.8, 48.9, 32.8, 19.6, 2.1, 2.6, 5.8, 24.0],
    "Newspaper": [69.2, 45.1, 69.3, 58.5, 58.4, 75.0, 23.5, 11.6, 1.0, 21.2, 24.2, 4.0],
    "Sales": [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8, 10.6, 8.6, 17.4]
}

df = pd.DataFrame(data)

x = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("R2: ", r2_score(y_test, y_pred))
print("Mean_squared_error: ", mean_squared_error(y_test, y_pred))
print("Mean_absolute_error: ", mean_absolute_error(y_test, y_pred))
print("Model", model.coef_)
print("Intercept", model.intercept_)