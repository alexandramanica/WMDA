# Exercițiul: Folosiți setul 20 Newsgroups din scikit-learn.
# Clasificați articolele în 3 categorii: 'sci.space', 'sci.med', 'talk.politics.guns'
#
# 1. Încărcați datele cu fetch_20newsgroups (subset='all')
#    și eliminați headerele, footerele și quote-urile
# 2. Vectorizați textul cu TfidfVectorizer(max_features=2000, ngram_range=(1,2))
# 3. Antrenați LogisticRegression și MultinomialNB
# 4. Afișați accuracy și classification_report pentru ambele
# 5. Afișați top 10 cuvinte distinctive pentru fiecare categorie
#    (din coeficienții LogisticRegression)

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

categories = ['sci.space', 'sci.med', 'talk.politics.guns']
data = fetch_20newsgroups(subset='all', categories=categories,
                          remove=('headers', 'footers', 'quotes'))

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

vector = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1,2))
#max-features ajuta la extragerea unui top de cuvinye

x_train_vect =  vector.fit_transform(x_train)
x_test_vect = vector.transform(x_test)

log_model = LogisticRegression()
log_model.fit(x_train_vect, y_train)
y_pred_log = log_model.predict(x_test_vect)

print("Accuracy " , accuracy_score(y_test, y_pred_log))
print("Classifcation", classification_report(y_test, y_pred_log))

keywords = vector.get_feature_names_out()

nb_model = MultinomialNB()
nb_model.fit(x_train_vect, y_train)
y_pred_nb = nb_model.predict(x_test_vect)

print("Accuracy " , accuracy_score(y_test, y_pred_nb))
print("Classifcation", classification_report(y_test, y_pred_nb))

print(keywords[:10])