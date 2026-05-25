### Cerinta

# Folositi un subset din `20 Newsgroups`, cu doua categorii:
# `sci.space` si `rec.sport.hockey`.
#
# 1. Impartiti textele in train si test cu `stratify=y`.
# 2. Transformati textele in vectori numerici cu `CountVectorizer`.
# 3. Antrenati un model `MultinomialNB`.
# 4. Afisati acuratetea pe setul de test.
# 5. Afisati primele 5 predictii impreuna cu eticheta reala.

### Cod De Pornire

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

categories = ["sci.space", "rec.sport.hockey"]

data = fetch_20newsgroups(subset="all", categories=categories, remove=("headers", "footers", "quotes"))

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

vector = CountVectorizer(stop_words="english")

x_train_vect = vector.fit_transform(x_train)
x_test_vect = vector.transform(x_test)

model = MultinomialNB()
model.fit(x_train_vect, y_train)

y_pred = model.predict(x_test_vect)

acc_score = accuracy_score(y_test, y_pred)

print(acc_score)
print(y_pred[:5])


