### Cerinta
#
# Generati date pentru relatia dintre viteza unei masini si distanta de franare.
#
# 1. Antrenati un model polinomial cu `degree=2`.
# 2. Antrenati un al doilea model cu `degree=9`.
# 3. Desenati pe acelasi grafic:
#
# ```text
# punctele observate
# curba degree=2
# curba degree=9
# ```


### Cod De Pornire

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(42)
speeds = np.linspace(10, 100, 20)
true_distance = 0.02 * speeds**2 - 1.5 * speeds + 50
noise = np.random.normal(0, 20, size=len(speeds))
braking_distance = true_distance + noise

# === START ===
df = pd.DataFrame({
    "speeds": speeds,
    "true_distance": true_distance,
    "braking_distance": braking_distance
})

x = df[["speeds"]]
y = df["braking_distance"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model_pipeline_degree2 = Pipeline([
    ("polynomial_features", PolynomialFeatures(degree=2)),
    ("linear_regression", LinearRegression())
])

model_pipeline_degree9 = Pipeline([
    ("polynomial_features", PolynomialFeatures(degree=9)),
    ("linear_regression", LinearRegression())
])

model_pipeline_degree2.fit(x_train, y_train)
model_pipeline_degree9.fit(x_train, y_train)

# Valori ordonate pentru desenarea curbelor
x_plot = np.linspace(10, 100, 300).reshape(-1, 1)

y_plot_degree2 = model_pipeline_degree2.predict(x_plot)
y_plot_degree9 = model_pipeline_degree9.predict(x_plot)

plt.figure(figsize=(10, 6))

# punctele observate
plt.scatter(x, y, label="Puncte observate")

# curba degree=2
plt.plot(x_plot, y_plot_degree2, label="Curba degree=2")

# curba degree=9
plt.plot(x_plot, y_plot_degree9, label="Curba degree=9")

plt.xlabel("Viteza mașinii")
plt.ylabel("Distanța de frânare")
plt.title("Regresie polinomială: viteza vs distanța de frânare")
plt.legend()
plt.grid(True)
plt.show()

# === END ===