# Exercițiul 2: Creați un model de clasificare text folosind un subset din 20 Newsgroups,
# cu două clase: 'sci.space' și 'rec.sport.hockey'. Vectorizați textul, antrenați modelul și
# calculați acuratețea pe setul de test.


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === START ===

categories = ["sci.space", "rec.sport.hockey"]

data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

x = data.data  # textul
y = data.target  # etichete

# stratify pastreaza proportie pt categorii
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

# CountVectorizer transformă textele în vectori numerici, adică într-un tabel cu frecvențe de cuvinte.
vectorizer = CountVectorizer(stop_words='english')

# Train: fit_transform = învață vocabularul + transformă
# Test: transform = folosește vocabularul din train + transformă
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

# Alegerea modelului
model = MultinomialNB()
model.fit(x_train_vectorized, y_train)

y_pred = model.predict(x_test_vectorized)

accuracy_score = accuracy_score(y_test, y_pred)

print(accuracy_score)
# === END ===
