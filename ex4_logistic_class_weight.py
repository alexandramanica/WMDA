# Generati un set de date in care aproximativ 92% dintre cazuri sunt sanatoase
# si 8% sunt cazuri de boala.
#
# 1. Antrenati un model `LogisticRegression` fara `class_weight`.
# 2. Antrenati un al doilea model cu `class_weight="balanced"`.
# 3. Afisati pentru ambele modele:
# confusion_matrix
# classification_report

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# 0.92, 0.8 sunt indicii care se iau din set

x, y = make_classification(
    n_samples=1200,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    weights=[0.92, 0.08],
    random_state=42
)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

model_balanced = LogisticRegression(class_weight="balanced", random_state=42)
model_balanced.fit(x_train, y_train)

y_pred_balanced = model_balanced.predict(x_test)

conf_matrix = confusion_matrix(y_test, y_pred_balanced)
class_report = classification_report(y_test, y_pred_balanced)

print(conf_matrix)
print(class_report)

model_unbalenced = LogisticRegression(random_state=42)
model_unbalenced.fit(x_train, y_train)

y_pred_unbalanced = model_unbalenced.predict(x_test)


conf_matrix_unbalanced = confusion_matrix(y_test, y_pred_unbalanced)
class_report_unbalanced = classification_report(y_test, y_pred_unbalanced)

print(conf_matrix_unbalanced)
print(class_report_unbalanced)