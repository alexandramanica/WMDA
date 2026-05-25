# Exercițiul 1: Extrageți titlul, prețul și disponibilitatea cărților de pe pagina principală
# a site-ului http://books.toscrape.com. Normalizați prețurile (float) și disponibilitatea
# (1 pentru "In stock", 0 altfel), apoi stocați-le într-un DataFrame pandas.

import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "http://books.toscrape.com/"

response = requests.get(url)

# === START ===
html_text = response.text

bs_parser = BeautifulSoup(html_text, 'html.parser')

data = []

for article in bs_parser.select("article.product_pod"):
    availability_norm = 0

    title = article.select_one("h3 a")["title"]
    product_price = article.select_one(".price_color").get_text(strip=True)
    product_price_float = float(product_price.replace("£", "").replace("Â", ""))
    availability = article.select_one(".instock.availability").get_text(strip=True)
    if availability == "In stock":
        availability_norm = 1
    rating = article.select_one("p.star-rating")["class"][1]

    data.append({
        "title": title,
        "product_price": product_price_float,
        "availability": availability_norm
    })

df = pd.DataFrame(data)

print(df)
# === END ===
