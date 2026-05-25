# Exercițiul 2: Creați un model de clasificare text pentru categoriile 'alt.atheism'
# și 'soc.religion.christian' din 20 Newsgroups. Folosiți CountVectorizer, antrenați
# un model Naive Bayes și afișați raportul de clasificare pe setul de test.

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

categories = ['alt.atheism', 'soc.religion.christian']
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

# === Your code starts here ===
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

vectorier = CountVectorizer(stop_words='english')

x_train_vectorized = vectorier.fit_transform(x_train)
x_test_vectorized = vectorier.transform(x_test)

model = MultinomialNB()
model.fit(x_train_vectorized, y_train)

y_pred = model.predict(x_test_vectorized)

report = classification_report(y_test, y_pred, target_names=data.target_names)

print(report)
# === Your code ends here ===
