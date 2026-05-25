"""Cursul 2 - clasificare: toate exemplele intr-un singur fisier comentat.

Acest fisier reuneste exemplele din directorul ``c2`` al repo-ului WMDA:
example1.py, example2.py, example3.py, example4.py, example5.py,
example6.py, example7.py, example7a.py si example8.py.

Ideea comuna tuturor exemplelor:

    date -> X si y -> train/test -> model.fit() -> model.predict() -> evaluare

Rulare:
    python c2_exercitii_clasificare_comentate.py

Pentru a afisa arborele grafic din exercitiul 4, apeleaza:
    exercitiul_4_arbore_decizie(show_plot=True)
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine, make_classification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


def titlu(numar, descriere):
    """Afiseaza un separator ca rezultatele exercitiilor sa fie usor de citit."""
    print("\n" + "=" * 72)
    print(f"EXERCITIUL {numar}: {descriere}")
    print("=" * 72)


def exercitiul_1_regresie_logistica():
    """Prezice daca un client cumpara, pe baza varstei si venitului."""
    titlu("1", "Regresie logistica - clientul cumpara sau nu")

    # Coloana ``purchased`` este target-ul: 0 = nu cumpara, 1 = cumpara.
    data = {
        "age": [25, 40, 35, 50, 28, 60, 45],
        "income": [50000, 70000, 60000, 80000, 52000, 100000, 75000],
        "purchased": [0, 1, 0, 1, 0, 1, 1],
    }
    df = pd.DataFrame(data)

    # X = informatiile de intrare; y = raspunsul care trebuie prezis.
    X = df[["age", "income"]]
    y = df["purchased"]

    # Modelul invata pe train si este verificat pe date nevazute din test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Desi se numeste "regresie", LogisticRegression este clasificator aici.
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Date initiale:")
    print(df)
    print(f"\nTest Accuracy: {accuracy:.2f}")


def exercitiul_2_clasificare_text():
    """Clasifica recenzii ca pozitive sau negative folosind frecventa cuvintelor."""
    titlu("2", "Clasificare text - recenzie pozitiva sau negativa")

    # 1 = recenzie pozitiva, 0 = recenzie negativa.
    data = [
        ("I absolutely loved this movie, it was fantastic!", 1),
        ("Horrible plot and terrible acting, wasted my time.", 0),
        ("An instant classic, superb in every aspect!", 1),
        ("I wouldn't recommend this film to anyone.", 0),
        ("It was just okay, nothing special or groundbreaking.", 0),
        ("Brilliant! I enjoyed every minute of it!", 1),
    ]
    df = pd.DataFrame(data, columns=["text", "label"])

    # CountVectorizer transforma textele intr-un tabel de frecvente:
    # fiecare coloana este un cuvant, iar fiecare valoare este numarul aparitiilor.
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    # Aceasta este ordinea din curs. In proiecte reale, se face intai split,
    # apoi fit_transform doar pe train si transform pe test, ca sa evitam
    # invatarea vocabularului din datele de test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # MultinomialNB este potrivit pentru frecvente de cuvinte.
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    comparison = pd.DataFrame(
        {
            "Review": df["text"].iloc[y_test.index],
            "Actual Label": y_test,
            "Predicted Label": y_pred,
        }
    )
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nPredictii comparate cu raspunsurile reale:")
    print(comparison)


def exercitiul_3_svm_si_dummies():
    """Prezice cumpararea folosind SVM si codificarea unei categorii."""
    titlu("3", "SVM - transformarea categoriei gender cu get_dummies")

    data = [
        (25, "Male", 50000, 0),
        (40, "Female", 70000, 1),
        (35, "Female", 60000, 0),
        (50, "Male", 80000, 1),
        (28, "Male", 52000, 0),
        (60, "Female", 100000, 1),
        (45, "Male", 75000, 1),
        (22, "Female", 48000, 0),
        (39, "Female", 68000, 1),
    ]
    df = pd.DataFrame(data, columns=["age", "gender", "income", "purchased"])

    # Modelul nu poate folosi direct "Male"/"Female".
    # Cu drop_first=True, Female devine cazul 0 si Male este reprezentat prin 1.
    df_encoded = pd.get_dummies(df, columns=["gender"], drop_first=True)

    X = df_encoded[["age", "income", "gender_Male"]]
    y = df_encoded["purchased"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # SVM liniar cauta o granita liniara intre clasa 0 si clasa 1.
    # In practica, pentru SVM ar fi bine sa standardizam age si income.
    model = SVC(kernel="linear", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    coefficients = pd.DataFrame(
        {"Feature": X_train.columns, "Coefficient": model.coef_[0]}
    )
    print("Date dupa one-hot encoding:")
    print(df_encoded)
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nCoeficientii SVM liniar:")
    print(coefficients)
    print("\nIntercept (bias):", model.intercept_[0])


def exercitiul_4_arbore_decizie(show_plot=False):
    """Prezice aprobarea creditului cu reguli invatate de un arbore."""
    titlu("4", "Arbore de decizie - aprobarea unui credit")

    data = [
        (50000, 700, 0.30, 1),
        (30000, 600, 0.40, 0),
        (80000, 750, 0.20, 1),
        (40000, 580, 0.50, 0),
        (75000, 720, 0.35, 1),
        (28000, 550, 0.45, 0),
        (90000, 780, 0.15, 1),
        (32000, 600, 0.42, 0),
        (66000, 710, 0.38, 1),
        (25000, 530, 0.50, 0),
    ]
    df = pd.DataFrame(
        data, columns=["income", "credit_score", "debt_ratio", "loan_approved"]
    )

    # Toate feature-urile sunt numerice; nu avem categorii pentru get_dummies.
    # loan_approved este target-ul si de aceea nu intra in X.
    X = df[["income", "credit_score", "debt_ratio"]]
    y = df["loan_approved"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # max_depth limiteaza complexitatea arborelui si riscul de overfitting.
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Afisarea graficului este optionala pentru ca scriptul complet sa ruleze
    # fara sa opreasca executia asteptand inchiderea unei ferestre.
    if show_plot:
        plt.figure(figsize=(8, 6))
        plot_tree(
            model,
            feature_names=["income", "credit_score", "debt_ratio"],
            class_names=["respins", "aprobat"],
            filled=True,
        )
        plt.title("Arborele de decizie pentru aprobarea creditului")
        plt.show()


def exercitiul_5_regularizare_l1_l2():
    """Compara regularizarea L1 cu L2 in regresia logistica."""
    titlu("5", "Regularizare - L1 versus L2")

    # Date sintetice: 10 coloane, dar numai unele contin informatie utila.
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # L1 poate duce coeficienti exact la zero: poate elimina feature-uri.
    model_l1 = LogisticRegression(
        penalty="l1", solver="saga", max_iter=1000, random_state=42
    )
    model_l1.fit(X_train, y_train)

    # L2 reduce coeficientii, dar de obicei ii pastreaza nenuli.
    model_l2 = LogisticRegression(
        penalty="l2", solver="saga", max_iter=1000, random_state=42
    )
    model_l2.fit(X_train, y_train)

    coef_comparison = pd.DataFrame(
        {
            "Feature": feature_names,
            "L1_Coefficient": model_l1.coef_[0],
            "L2_Coefficient": model_l2.coef_[0],
        }
    )
    non_zero_features = coef_comparison[
        coef_comparison["L1_Coefficient"] != 0
    ]["Feature"].tolist()

    print("Comparatia coeficientilor:")
    print(coef_comparison)
    print(f"\nTest Accuracy - L1: {model_l1.score(X_test, y_test):.2f}")
    print(f"Test Accuracy - L2: {model_l2.score(X_test, y_test):.2f}")
    print("\nFeature-uri pastrate de L1 (coeficient nenul):")
    print(non_zero_features)


def exercitiul_6_metrici():
    """Arata cum citim confusion matrix, precision, recall si accuracy."""
    titlu("6", "Metrici - confusion matrix, precision, recall, accuracy")

    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Pentru clasificare binara, matricea este:
    # [[TN, FP],
    #  [FN, TP]]
    # FN este un caz pozitiv real pe care modelul l-a ratat.
    conf_mat = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(conf_mat)
    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")


def exercitiul_7_clase_dezechilibrate():
    """Compara clasificarea medicala cu si fara ponderarea clasei rare."""
    titlu("7", "Clase dezechilibrate - boala rara")

    # Clasa 0 = sanatos (90%), clasa 1 = bolnav (10%).
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        weights=[0.90, 0.10],
        random_state=42,
    )

    # stratify=y este o completare practica utila: mentine proportiile claselor
    # similare in train si test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # balanced da o penalizare mai mare greselilor asupra clasei rare (bolnav).
    model_balanced = LogisticRegression(class_weight="balanced", random_state=42)
    model_balanced.fit(X_train, y_train)
    y_pred_balanced = model_balanced.predict(X_test)

    print("Cu class_weight='balanced':")
    print(confusion_matrix(y_test, y_pred_balanced))
    print(
        classification_report(y_test, y_pred_balanced, zero_division=0)
    )

    # Fara ponderare, modelul poate favoriza clasa majoritara: sanatos.
    model_no_weight = LogisticRegression(random_state=42)
    model_no_weight.fit(X_train, y_train)
    y_pred_no_weight = model_no_weight.predict(X_test)

    print("Fara class_weight:")
    print(confusion_matrix(y_test, y_pred_no_weight))
    print(classification_report(y_test, y_pred_no_weight, zero_division=0))


def exercitiul_7a_grid_search():
    """Alege automat adancimea arborelui pe datele Wine."""
    titlu("7a", "GridSearchCV - alegerea lui max_depth")

    wine = load_wine()
    X = wine.data
    y = wine.target

    # Wine are trei clase. stratify pastreaza aproximativ proportiile claselor.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # max_depth este hiperparametru: il stabilim inainte de modelul final.
    param_grid = {"max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10]}

    # Pentru fiecare adancime, cv=5 face cinci validari in interiorul train-ului.
    # Setul X_test ramane neatins pana la evaluarea finala.
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("Best Parameters:", grid_search.best_params_)
    print(f"Best Cross-Val Score: {grid_search.best_score_:.3f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")


def exercitiul_8_detectarea_fraudei():
    """Detecteaza fraude rare folosind regresie logistica ponderata."""
    titlu("8", "Aplicatie finala - detectarea fraudelor bancare")

    # Clasa 0 = tranzactie legitima (aprox. 99%).
    # Clasa 1 = tranzactie frauduloasa (aprox. 1%).
    X, y = make_classification(
        n_samples=2000,
        n_features=6,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.99, 0.01],
        random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Pentru o clasa atat de rara, stratify ajuta ca si testul sa contina fraude.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Accuracy singura ar fi inselatoare: un model care spune mereu "legitim"
    # ar avea aproape 99% accuracy, dar nu ar prinde fraude.
    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    coefficients = pd.DataFrame(
        {"Feature": feature_names, "Coefficient": model.coef_[0]}
    )
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Coeficientii regresiei logistice:")
    print(coefficients)


def ruleaza_toate_exercitiile():
    """Ruleaza exercitiile in ordinea in care apar in curs."""
    exercitiul_1_regresie_logistica()
    exercitiul_2_clasificare_text()
    exercitiul_3_svm_si_dummies()
    exercitiul_4_arbore_decizie()
    exercitiul_5_regularizare_l1_l2()
    exercitiul_6_metrici()
    exercitiul_7_clase_dezechilibrate()
    exercitiul_7a_grid_search()
    exercitiul_8_detectarea_fraudei()


if __name__ == "__main__":
    ruleaza_toate_exercitiile()
