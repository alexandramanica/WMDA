# ## Exercitiul 7: Ore De Studiu Si Nota
#
# ### Cerinta
#
# Construiti un model de regresie liniara care estimeaza nota unui student
# in functie de numarul de ore studiate.
#
# 1. Antrenati modelul.
# 2. Afisati coeficientul si interceptul.
# 3. Preziceti nota pentru un student care a studiat `7` ore.
# 4. Desenati punctele si dreapta de regresie.

### Cod De Pornire

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

hours_studied = np.array([1, 2, 3, 4, 5, 6])
exam_score = np.array([48, 55, 63, 68, 78, 87])

# === START ===

# X trebuie sa fie bidimensional pentru scikit-learn:
# 6 observatii si o singura caracteristica, numarul de ore studiate.
x = hours_studied.reshape(-1, 1)
y = exam_score

# 1. Antrenarea modelului
model = LinearRegression()
model.fit(x, y)

# 2. Afisarea coeficientului si interceptului
coefficient = model.coef_[0]
intercept = model.intercept_

print("Coeficient:", coefficient)
print("Intercept:", intercept)

# 3. Predictia notei pentru 7 ore studiate
new_hours = np.array([[7]])
predicted_score = model.predict(new_hours)

print("Nota estimata pentru 7 ore studiate:", predicted_score[0])

# 4. Predictii pentru dreapta de regresie
y_pred = model.predict(x)

# Desenarea punctelor observate
plt.scatter(hours_studied, exam_score, label="Note reale")

# Desenarea dreptei invatate de model
plt.plot(hours_studied, y_pred, label="Dreapta de regresie")

# Marcarea predictiei pentru 7 ore
plt.scatter(7, predicted_score[0], label="Predictie pentru 7 ore")

plt.xlabel("Ore studiate")
plt.ylabel("Nota")
plt.title("Regresie liniara: ore studiate si nota")
plt.legend()
plt.show()

# === END ===