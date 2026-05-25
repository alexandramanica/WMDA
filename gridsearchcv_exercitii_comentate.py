"""GridSearchCV - exercitii comentate pentru situatiile importante.

Fisierul continua exemplele din cursul 2 despre clasificare. Scopul sau este
sa arate CAND si CUM folosim GridSearchCV, nu doar sa memoram sintaxa.

Ce face GridSearchCV:
    1. primeste un model si o lista de hiperparametri;
    2. incearca toate combinatiile din grila;
    3. evalueaza fiecare combinatie prin cross-validation;
    4. pastreaza configuratia cu scorul cerut cel mai bun;
    5. reantreneaza automat cel mai bun model pe tot setul de train.

Regula esentiala:
    Setul de test NU se foloseste pentru alegerea hiperparametrilor.
    Il folosim o singura data, dupa ce GridSearchCV a ales modelul.

Rulare:
    python gridsearchcv_exercitii_comentate.py
"""

import pandas as pd
from sklearn.datasets import load_diabetes, load_wine, make_classification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    make_scorer,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def titlu(numar, text):
    """Afiseaza separat fiecare situatie de utilizare."""
    print("\n" + "=" * 78)
    print(f"SITUATIA {numar}: {text}")
    print("=" * 78)


def afiseaza_top_rezultate(grid_search, nr_randuri=5):
    """Afiseaza cele mai bune combinatii incercate in cross-validation."""
    rezultate = pd.DataFrame(grid_search.cv_results_)
    coloane = ["params", "mean_test_score", "rank_test_score"]
    top = rezultate.sort_values("rank_test_score")[coloane].head(nr_randuri)
    print("\nCele mai bune configuratii testate:")
    print(top.to_string(index=False))


def situatia_1_arbore_un_parametru():
    """Cazul de baza: alegem max_depth pentru un arbore de decizie."""
    titlu("1", "Arbore de decizie - cautam max_depth")

    wine = load_wine()
    X = wine.data
    y = wine.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Folosim GridSearch cand nu stim cat de complex trebuie sa fie arborele.
    # Prea mic: underfitting. Prea mare: risc de overfitting.
    param_grid = {
        "max_depth": [2, 3, 4, 5, None],
    }

    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.best_estimator_.predict(X_test)

    print("Cand se foloseste: cand alegem complexitatea unui arbore.")
    print("Best Parameters:", grid_search.best_params_)
    print(f"Best Cross-Val Accuracy: {grid_search.best_score_:.3f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    afiseaza_top_rezultate(grid_search)


def situatia_2_regresie_logistica_pipeline():
    """Cautam regularizarea dupa ce standardizam variabilele numerice."""
    titlu("2", "Regresie logistica - alegem regularizarea intr-un Pipeline")

    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Pipeline tine impreuna preprocesarea si modelul.
    # Important: scaler-ul este invatat separat in fiecare fold de CV,
    # deci nu "vede" datele de validare inainte de evaluare.
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )

    # C controleaza forta regularizarii:
    # C mic = regularizare mai puternica; C mare = regularizare mai slaba.
    # Pentru penalitati diferite folosim grile compatibile cu solver-ul.
    param_grid = [
        {
            "model__penalty": ["l1"],
            "model__solver": ["liblinear"],
            "model__C": [0.01, 0.1, 1, 10],
        },
        {
            "model__penalty": ["l2"],
            "model__solver": ["liblinear"],
            "model__C": [0.01, 0.1, 1, 10],
        },
    ]

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)

    print("Cand se foloseste: cand modelul are hiperparametri de regularizare.")
    print("Observa sintaxa model__C: parametrul apartine pasului 'model'.")
    print("Best Parameters:", grid_search.best_params_)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    afiseaza_top_rezultate(grid_search)


def situatia_3_svm_kernel_si_c():
    """Cautam tipul granitei SVM si forta penalizarii."""
    titlu("3", "SVM - cautam kernel, C si gamma")

    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data,
        wine.target,
        test_size=0.25,
        random_state=42,
        stratify=wine.target,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC()),
        ]
    )

    # Pentru kernel='linear' nu avem nevoie de gamma.
    # Pentru kernel='rbf', gamma controleaza forma granitei curbate.
    param_grid = [
        {
            "svc__kernel": ["linear"],
            "svc__C": [0.1, 1, 10],
        },
        {
            "svc__kernel": ["rbf"],
            "svc__C": [0.1, 1, 10],
            "svc__gamma": ["scale", 0.01, 0.1],
        },
    ]

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)

    print("Cand se foloseste: cand nu stim daca separarea este liniara sau curba.")
    print("Best Parameters:", grid_search.best_params_)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    afiseaza_top_rezultate(grid_search)


def situatia_4_text_cu_pipeline():
    """Cautam simultan cum vectorizam textul si cum configuram clasificatorul."""
    titlu("4", "Clasificare text - CountVectorizer si MultinomialNB")

    texte = [
        "space shuttle mission launches into orbit",
        "nasa plans a new moon mission",
        "astronaut works on the space station",
        "rocket launch sends satellite into orbit",
        "telescope observes planets and distant stars",
        "mars rover collects samples on the planet",
        "spacecraft travels beyond the moon",
        "scientists study a galaxy using a telescope",
        "satellite launch was successful today",
        "astronaut training prepares crews for space",
        "the hockey player scored a winning goal",
        "the nhl team won the playoff game",
        "goalie saved the puck during overtime",
        "hockey fans celebrated the championship",
        "the coach changed lines before the match",
        "a penalty gave the team a power play",
        "the puck crossed the goal line",
        "the ice hockey season begins tonight",
        "the goalkeeper stopped three shots",
        "the team practiced skating and passing",
    ]
    etichete = [
        "sci.space",
        "sci.space",
        "sci.space",
        "sci.space",
        "sci.space",
        "sci.space",
        "sci.space",
        "sci.space",
        "sci.space",
        "sci.space",
        "rec.sport.hockey",
        "rec.sport.hockey",
        "rec.sport.hockey",
        "rec.sport.hockey",
        "rec.sport.hockey",
        "rec.sport.hockey",
        "rec.sport.hockey",
        "rec.sport.hockey",
        "rec.sport.hockey",
        "rec.sport.hockey",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        texte, etichete, test_size=0.3, random_state=42, stratify=etichete
    )

    # Pipeline-ul este foarte important la text:
    # CountVectorizer isi invata vocabularul numai din fold-ul de antrenare.
    pipeline = Pipeline(
        steps=[
            ("vectorizer", CountVectorizer()),
            ("model", MultinomialNB()),
        ]
    )

    # Cautam parametri atat pentru transformarea textului, cat si pentru model.
    param_grid = {
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "vectorizer__stop_words": [None, "english"],
        "model__alpha": [0.1, 1.0],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)

    print("Cand se foloseste: cand si vectorizarea are alegeri de facut.")
    print("Best Parameters:", grid_search.best_params_)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nTexte de test si predictii:")
    for text, real, prezis in zip(X_test, y_test, y_pred):
        print(f"- {text!r} -> real={real}, prezis={prezis}")
    afiseaza_top_rezultate(grid_search)


def situatia_5_clasa_rara_scorul_corect():
    """Alegem modelul dupa recall, nu dupa accuracy, cand cazul rar conteaza."""
    titlu("5", "Frauda/boala rara - GridSearchCV optimizat pentru recall")

    X, y = make_classification(
        n_samples=2000,
        n_features=8,
        n_informative=4,
        n_redundant=1,
        weights=[0.97, 0.03],
        flip_y=0.01,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__class_weight": [None, "balanced"],
    }

    # Putem calcula mai multe metrici in aceeasi cautare.
    # refit='recall' spune: modelul final ales este cel mai bun la gasirea
    # cazurilor pozitive reale, nu neaparat cel cu accuracy cea mai mare.
    # zero_division=0 trateaza cazul in care un model nu prezice nicio frauda.
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
    }
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        refit="recall",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)

    print("Cand se foloseste: la frauda sau boala, unde un caz pozitiv ratat doare.")
    print("Modelul final este ales dupa recall, nu dupa accuracy.")
    print("Best Parameters:", grid_search.best_params_)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Test Recall pentru clasa rara: {recall_score(y_test, y_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    rezultate = pd.DataFrame(grid_search.cv_results_)
    coloane = [
        "params",
        "mean_test_accuracy",
        "mean_test_precision",
        "mean_test_recall",
        "rank_test_recall",
    ]
    print("Comparatie metrici in cross-validation:")
    print(
        rezultate.sort_values("rank_test_recall")[coloane]
        .head(5)
        .to_string(index=False)
    )


def situatia_6_regresie_bonus():
    """GridSearchCV nu este numai pentru clasificare: il folosim si la regresie."""
    titlu("6 (BONUS)", "Regresie - alegem arborele pentru o tinta numerica")

    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.2, random_state=42
    )

    # Aici y este numeric (progresia bolii), deci problema este regresie.
    # Nu folosim accuracy; folosim scoruri potrivite regresiei.
    param_grid = {
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_leaf": [1, 5, 10],
    }
    grid_search = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)

    # Scikit-learn foloseste minus MAE in cautare deoarece "mai mare" trebuie
    # sa insemne "mai bun". La afisare calculam MAE normal, pozitiv.
    print("Cand se foloseste: si la predictii numerice, nu numai la clase.")
    print("Best Parameters:", grid_search.best_params_)
    print(f"Test MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"Test R2: {r2_score(y_test, y_pred):.3f}")
    afiseaza_top_rezultate(grid_search)


def ruleaza_toate_situatiile():
    """Ruleaza exemplele rezolvate in ordine, de la simplu la aplicat."""
    situatia_1_arbore_un_parametru()
    situatia_2_regresie_logistica_pipeline()
    situatia_3_svm_kernel_si_c()
    situatia_4_text_cu_pipeline()
    situatia_5_clasa_rara_scorul_corect()
    situatia_6_regresie_bonus()


if __name__ == "__main__":
    ruleaza_toate_situatiile()
