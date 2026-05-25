# Exercițiul: Descărcați datele despre pasagerii Titanic de la URL-ul de mai jos.
# Preziceți supraviețuirea (Survived) pe baza coloanelor: Pclass, Sex, Age, SibSp, Parch, Fare.
#
# 1. Curățați datele:
#    - înlocuiți valorile lipsă din 'Age' cu mediana
#    - codificați coloana 'Sex' cu pd.get_dummies
# 2. Antrenați un DecisionTreeClassifier(max_depth=4) și un SVC(kernel='rbf')
#    (nu uita scalarea unde e cazul)
# 3. Afișați accuracy și classification_report pentru ambele modele
# 4. Afișați feature_importances_ pentru DecisionTree

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score, classification_report

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

df = pd.get_dummies(df, columns=["Sex"], drop_first=True)
df["Age"] = df["Age"].fillna(df["Age"].median())
print(df)

x = df[["Pclass", "Sex_male", "Age", "SibSp", "Parch", "Fare"]]
y = df["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

model = Pipeline(
    [("scaler", StandardScaler()),
     ("svc", SVC(kernel="rbf", random_state=42))]
)

model_tree = DecisionTreeClassifier(max_depth=4, random_state=42)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

model_tree.fit(x_train, y_train)
y_pred_tree = model_tree.predict(x_test)

print("Accuracy score: ", accuracy_score(y_test, y_pred))
print("Classifaction report: ", classification_report(y_test, y_pred))

print("Accuracy score tree: ", accuracy_score(y_test, y_pred_tree))
print("Classifaction report tree: ", classification_report(y_test, y_pred_tree))