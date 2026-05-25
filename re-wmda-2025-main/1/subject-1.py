# Exercițiul 1: Accesați pagina http://quotes.toscrape.com și extrageți toate citatele,
# autorii și tag-urile de pe prima pagină. Salvați rezultatul într-un DataFrame pandas
# cu coloanele: 'quote', 'author', 'tags' (tags este o listă de șiruri).

import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "http://quotes.toscrape.com"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
data = []

for quote_card in soup.select(".quote"):
    quote = quote_card.select_one(".text").get_text(strip=True)
    author = quote_card.select_one(".author").get_text(strip=True)
    tags = quote_card.select_one(".keywords")["content"]

    data.append({
        "quote": quote,
        "author": author,
        "tags": tags
    })

df = pd.DataFrame(data)

print(data)