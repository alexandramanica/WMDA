# Folositi setul de date `Diabetes` pentru a compara doua modele de regresie:
# `Ridge` si `Lasso`.

# 1. Impartiti datele in train si test.
# 2. Antrenati un model `Ridge(alpha=1.0)`.
# 3. Antrenati un model `Lasso(alpha=1.0)`.
# 4. Afisati coeficientii celor doua modele.
# 5. Afisati `R2` si `MSE` pentru fiecare model.
# 6. Verificati daca `Lasso` produce coeficienti egali cu zero.
#
# ### Cod De Pornire
#

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

data = load_diabetes()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model_ridge = Ridge(alpha=1.0)
model_ridge.fit(x_train, y_train)
y_ridge_pred = model_ridge.predict(x_test)
print("r2_score ridge", r2_score(y_test, y_ridge_pred))
print("mse_score ridge", mean_squared_error(y_test, y_ridge_pred))
print("ridge coeff", model_ridge.coef_)


model_lasso = Lasso(alpha=1.0)
model_lasso.fit(x_train, y_train)
y_lasson_pred = model_lasso.predict(x_test)
print("r2_score lasso", r2_score(y_test, y_lasson_pred))
print("mse_score lasso", mean_squared_error(y_test, y_lasson_pred))
print("lasso coeff", model_lasso.coef_)