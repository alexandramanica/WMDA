# Creati un model care prezice daca un client cumpara un produs (`purchased`),
# folosind varsta (`age`), venitul (`income`) si tipul de abonament
# (`subscription`: `basic` sau `premium`).


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    "age": [22, 25, 29, 31, 35, 38, 42, 45, 50, 53, 57, 60],
    "income": [28000, 32000, 36000, 41000, 47000, 52000, 59000, 65000, 72000, 78000, 85000, 92000],
    "subscription": ["basic", "basic", "premium", "basic", "premium", "premium",
                     "basic", "premium", "premium", "basic", "premium", "premium"],
    "purchased": [0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# === START ===
df_encoded = pd.get_dummies(df, columns=["subscription"], drop_first=True)

x = df_encoded[["age", "income", "subscription_premium"]]
y = df["purchased"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

acc_score = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(acc_score)
print(conf_matrix)
# === END ===
