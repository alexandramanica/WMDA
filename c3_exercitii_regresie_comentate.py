"""Cursul 3 - regresie: toate exemplele intr-un singur fisier comentat.

Acest fisier reuneste exemplele profesorului din directorul ``c3``:
example1.py, example2.py, example3.py, example4.py, example5.py,
example6.py, example7.py, example8.py si example9.py.

Codul exemplelor urmeaza pasii din curs. Comentariile in romana explica
rolul instructiunilor, fara sa schimbe exercitiile profesorului.

Ideea comuna regresiei:

    date -> X si y numeric -> model.fit() -> model.predict() -> evaluare

Diferenta fata de clasificare:

    clasificare: y este o clasa (0/1, spam/non-spam)
    regresie:    y este un numar (pret, nota, vanzari)

Rulare:
    python c3_exercitii_regresie_comentate.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor


def titlu(numar, descriere):
    """Afiseaza un separator pentru a identifica usor fiecare exercitiu."""
    print("\n" + "=" * 72)
    print(f"EXERCITIUL {numar}: {descriere}")
    print("=" * 72)


def exercitiul_1_pretul_casei():
    """Regresie liniara: estimarea pretului pe baza caracteristicilor casei."""
    titlu("1", "LinearRegression - pretul unei case")

    # Setul de date contine caracteristicile unei case si pretul sau.
    # price este valoarea numerica pe care modelul o va prezice.
    data = {
        "sqft": [1500, 2000, 1100, 2500, 1400, 2300],
        "bedrooms": [3, 4, 2, 5, 3, 4],
        "location": ["cityA", "cityB", "cityA", "cityB", "cityA", "cityB"],
        "price": [300000, 400000, 200000, 500000, 280000, 450000],
    }
    df = pd.DataFrame(data)

    # X = informatiile de intrare; y = target-ul numeric (pretul).
    X = df[["sqft", "bedrooms", "location"]]
    y = df["price"]

    # location este text si trebuie transformat in coloane numerice dummy.
    # Cu drop_first=True, cityA este cazul de baza, iar cityB devine coloana.
    X_encoded = pd.get_dummies(X, columns=["location"], drop_first=True)

    # LinearRegression invata o formula liniara pentru pret.
    model = LinearRegression()
    model.fit(X_encoded, y)

    # Coeficientii arata contributia fiecarei caracteristici in formula.
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

    # Prezicem pretul unei case noi: 1600 sqft, 3 dormitoare, in cityB.
    new_house = pd.DataFrame(
        {
            "sqft": [1600],
            "bedrooms": [3],
            "location": ["cityB"],
        }
    )

    # Casa noua este codificata si aliniata la aceleasi coloane ca X_encoded.
    new_house_encoded = pd.get_dummies(
        new_house, columns=["location"], drop_first=True
    )
    new_house_encoded = new_house_encoded.reindex(
        columns=X_encoded.columns, fill_value=0
    )

    predicted_price = model.predict(new_house_encoded)
    print("Predicted price for the new house:", predicted_price[0])


def exercitiul_2_ore_studiu_nota():
    """Regresie liniara simpla: o caracteristica si o dreapta de regresie."""
    titlu("2", "LinearRegression - ore studiate si nota la examen")

    # X va fi numarul de ore studiate, iar y este nota obtinuta.
    hours_studied = np.array([1, 2, 3, 4, 5, 6])
    exam_score = np.array([50, 60, 65, 70, 75, 90])

    # Scikit-learn cere ca X sa fie tabel 2D: 6 observatii, 1 coloana.
    X = hours_studied.reshape(-1, 1)
    y = exam_score

    # Modelul cauta dreapta: nota = intercept + slope * ore.
    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_
    print(f"Slope (Coefficient): {slope:.3f}")
    print(f"Intercept: {intercept:.3f}")

    # y_pred contine notele estimate de dreapta pentru orele observate.
    y_pred = model.predict(X)

    # Punctele reprezinta datele reale, iar linia este estimarea modelului.
    plt.scatter(hours_studied, exam_score, label="Data Points")
    plt.plot(hours_studied, y_pred, label="Best Fit Line")
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.title("Linear Regression: Hours Studied vs Exam Score")
    plt.legend()
    plt.show()


def exercitiul_3_regresie_polinomiala():
    """Regresie polinomiala: relatia curbata viteza-distanta de franare."""
    titlu("3", "Polynomial Regression - viteza si distanta de franare")

    # Relatia reala este de gradul 2, apoi se adauga zgomot aleator.
    np.random.seed(42)
    speeds = np.linspace(10, 100, 20)
    true_braking_distance = 0.02 * speeds**2 - 1.5 * speeds + 50
    noise = np.random.normal(loc=0.0, scale=20.0, size=len(speeds))
    braking_distance = true_braking_distance + noise

    # degree=2 creeaza coloanele 1, speed si speed^2.
    X = speeds.reshape(-1, 1)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    # LinearRegression poate invata o curba deoarece primeste termenul speed^2.
    model = LinearRegression()
    model.fit(X_poly, braking_distance)

    # Generam multe puncte pentru a desena curba estimata cat mai lin.
    speeds_plot = np.linspace(min(speeds), max(speeds), 100)
    speeds_plot_poly = poly.transform(speeds_plot.reshape(-1, 1))
    braking_distance_pred = model.predict(speeds_plot_poly)

    plt.scatter(speeds, braking_distance, label="Data (Observed)")
    plt.plot(speeds_plot, braking_distance_pred, label="Polynomial Fit")
    plt.xlabel("Car's Speed (km/h)")
    plt.ylabel("Braking Distance (m)")
    plt.title("Polynomial Regression: Speed vs. Braking Distance")
    plt.legend()
    plt.show()


def exercitiul_4_arbore_regresie():
    """DecisionTreeRegressor: estimeaza pretul folosind reguli cu praguri."""
    titlu("4", "DecisionTreeRegressor - pretul unei case")

    data = {
        "location": ["cityA", "cityA", "cityB", "cityB", "cityA", "cityB"],
        "rooms": [2, 3, 2, 4, 3, 5],
        "sqft": [800, 1200, 900, 1800, 1100, 2200],
        "price": [100000, 180000, 160000, 290000, 200000, 360000],
    }
    df = pd.DataFrame(data)

    X = df[["location", "rooms", "sqft"]]
    y = df["price"]

    # location devine o caracteristica numerica: location_cityB.
    X_encoded = pd.get_dummies(X, columns=["location"], drop_first=True)

    # Arborele de regresie produce o valoare numerica in frunzele sale.
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X_encoded, y)

    # Casa noua este deja reprezentata cu aceleasi coloane numerice.
    new_house = pd.DataFrame(
        {
            "rooms": [4],
            "sqft": [2000],
            "location_cityB": [1],
        }
    )

    predicted_price = tree_reg.predict(new_house)
    print("Predicted price for new house:", predicted_price[0])


def exercitiul_5_overfitting():
    """Arata cum un polinom de grad mare urmareste zgomotul din date."""
    titlu("5", "Overfitting - polinom de grad mare pe putine date")

    # Tendinta adevarata este simpla: o curba de gradul 2.
    # Datele observate primesc zgomot, asemenea datelor reale.
    np.random.seed(0)
    X_small = np.linspace(-3, 3, 8)
    y_true = 0.5 * X_small**2
    noise = np.random.normal(loc=0.0, scale=2.0, size=len(X_small))
    y_small = y_true + noise

    X_small = X_small.reshape(-1, 1)

    # Modelul are grad 9, desi relatia reala este doar de grad 2.
    # Astfel poate deveni prea flexibil pentru cele 8 puncte.
    poly = PolynomialFeatures(degree=9)
    X_poly = poly.fit_transform(X_small)

    model = LinearRegression()
    model.fit(X_poly, y_small)

    # Curba este afisata pe 200 de puncte pentru a vedea oscilatiile.
    X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot_pred = model.predict(X_plot_poly)

    plt.scatter(X_small, y_small, label="Noisy Data Points")
    plt.plot(X_plot, y_plot_pred, label="Degree=9 Polynomial Fit")
    plt.plot(X_plot, 0.5 * X_plot**2, label="True Underlying Trend (Quadratic)")
    plt.title("High-Degree Polynomial Overfitting Example")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def exercitiul_6_ridge_lasso():
    """Compara regularizarea L2 (Ridge) cu regularizarea L1 (Lasso)."""
    titlu("6", "Regularizare - Ridge versus Lasso")

    np.random.seed(42)

    # X are 100 de observatii si 5 feature-uri numerice.
    X = np.random.rand(100, 5) * 10

    # Acestia sunt coeficientii folositi pentru a construi y.
    # Feature-urile cu coeficient 0 nu contribuie la tinta reala.
    true_coefs = np.array([1.5, 0.0, -2.0, 0.0, 3.0])
    y = X.dot(true_coefs) + np.random.normal(0, 2, size=100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Ridge foloseste regularizare L2: micsoreaza coeficientii.
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_coefs = ridge.coef_
    ridge_intercept = ridge.intercept_

    # Lasso foloseste regularizare L1: poate face unii coeficienti zero.
    lasso = Lasso(alpha=1.0)
    lasso.fit(X_train, y_train)
    lasso_coefs = lasso.coef_
    lasso_intercept = lasso.intercept_

    print("True coefficients:", true_coefs)
    print("\nRidge coefficients:", ridge_coefs)
    print("Ridge intercept:", ridge_intercept)
    print("\nLasso coefficients:", lasso_coefs)
    print("Lasso intercept:", lasso_intercept)

    # score() pentru un model de regresie returneaza R^2.
    ridge_score = ridge.score(X_test, y_test)
    lasso_score = lasso.score(X_test, y_test)
    print(f"\nRidge R^2 on test data: {ridge_score:.3f}")
    print(f"Lasso R^2 on test data: {lasso_score:.3f}")


def exercitiul_7_metrici_regresie():
    """Calculeaza R2, MSE si MAE pentru preturi prezise deja."""
    titlu("7", "Metrici - R2, MSE si MAE")

    # Aici nu antrenam un model: avem direct valorile reale si prezise.
    actual_prices = np.array([300000, 450000, 250000, 400000, 320000])
    predicted_prices = np.array([280000, 480000, 230000, 420000, 310000])

    # R2 mai mare inseamna ca predictiile urmaresc mai bine variatia reala.
    # MSE si MAE mai mici inseamna erori mai mici.
    r2 = r2_score(actual_prices, predicted_prices)
    mse = mean_squared_error(actual_prices, predicted_prices)
    mae = mean_absolute_error(actual_prices, predicted_prices)

    print("R2 Score:", r2)
    print("MSE:", mse)
    print("MAE:", mae)


def exercitiul_8_publicitate_vanzari():
    """Flux complet: regresie liniara multipla, train/test si evaluare."""
    titlu("8", "LinearRegression - publicitate si vanzari")

    data = {
        "TV": [230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.8],
        "Radio": [37.8, 39.3, 45.9, 41.3, 10.8, 48.9, 32.8, 19.6, 2.1, 2.6],
        "Newspaper": [
            69.2,
            45.1,
            69.3,
            58.5,
            58.4,
            75.0,
            23.5,
            11.6,
            1.0,
            21.2,
        ],
        "Sales": [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8, 10.6],
    }
    df = pd.DataFrame(data)

    # Avem trei feature-uri numerice si un target numeric: Sales.
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("Coefficients (TV, Radio, Newspaper):", model.coef_)
    print("Intercept:", model.intercept_)
    print("R^2 on test set:", r2)
    print("MSE on test set:", mse)


def exercitiul_9_knn_regresie():
    """KNN pentru regresie: pretul este media preturilor vecinilor apropiati."""
    titlu("9", "KNeighborsRegressor - case asemanatoare")

    data = {
        "location": ["cityA", "cityB", "cityA", "cityB", "cityA", "cityB"],
        "sqft": [1000, 2000, 1200, 1800, 900, 2200],
        "bedrooms": [2, 4, 3, 4, 2, 5],
        "price": [200000, 400000, 260000, 350000, 180000, 450000],
    }
    df = pd.DataFrame(data)

    X = df[["location", "sqft", "bedrooms"]]
    y = df["price"]

    # location este transformata numeric pentru a fi folosita la distante.
    X_encoded = pd.get_dummies(X, columns=["location"], drop_first=True)

    # KNN compara distante; StandardScaler pune feature-urile pe scari similare.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # Pentru fiecare predicție, sunt utilizati cei mai apropiati 3 vecini.
    knn_reg = KNeighborsRegressor(n_neighbors=3)
    knn_reg.fit(X_train, y_train)

    y_pred_test = knn_reg.predict(X_test)
    print("Test Set Predictions:", y_pred_test)
    print("True Values:", y_test.values)

    new_house = pd.DataFrame(
        {
            "sqft": [1500],
            "bedrooms": [3],
            "location_cityB": [1],
        }
    )

    # Casa noua trebuie scalata folosind acelasi scaler.
    new_house_scaled = scaler.transform(new_house)

    predicted_price = knn_reg.predict(new_house_scaled)
    print("Predicted price for the new house:", predicted_price[0])


def ruleaza_toate_exercitiile():
    """Ruleaza exemplele c3 in ordinea materialului profesorului."""
    exercitiul_1_pretul_casei()
    exercitiul_2_ore_studiu_nota()
    exercitiul_3_regresie_polinomiala()
    exercitiul_4_arbore_regresie()
    exercitiul_5_overfitting()
    exercitiul_6_ridge_lasso()
    exercitiul_7_metrici_regresie()
    exercitiul_8_publicitate_vanzari()
    exercitiul_9_knn_regresie()


if __name__ == "__main__":
    ruleaza_toate_exercitiile()
