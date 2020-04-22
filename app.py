# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:50:13 2020

@author: Francine Mäkelä
"""
import requests
from bs4 import BeautifulSoup
import pandas
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef

base_url = "https://pokemondb.net/pokedex/"
legendary_url = "https://nintendo.fandom.com/wiki/Legendary_Pokémon"

def get_soup(path):
    r = requests.get(path)
    c = r.content
    return BeautifulSoup(c, "html.parser")

def get_type(row):
    types = row.find("td", {"class":"cell-icon"}).find_all(lambda a: a.name=='a' and a.has_attr('class'))
    ts = []
    for t in types:
        ts.append(t.text)
    return ts    

# Fetch all pokemons
soup = get_soup(base_url + "all")
poke_table = soup.find(lambda tag: tag.name=='table' and tag.has_attr('id') and tag['id']=="pokedex")
rows = poke_table.find_all(lambda tag: tag.name=='tr')

poke_list = []
type_list = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice",
                 "Fighting", "Poison", "Ground", "Flying", "Psychic",
                 "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]
for row in rows:
    p = []
    pokemon = {}
    try:
        pokemon["Name"] = row.find("a", {"class":"ent-name"}).text.title()
    except:
        continue
    
    
    pokemon["Id"] = row.find("span", {"class":"infocard-cell-data"}).text
    pokemon["Type"] = get_type(row)
    pokemon["Total"] = row.find("td", {"class":"cell-total"}).text
    
    other_numbers = row.find_all("td", {"class":"cell-num"})
    pokemon["HP"] = other_numbers[1].text
    pokemon["Attack"] = other_numbers[2].text
    pokemon["Defense"] = other_numbers[3].text
    pokemon["Sp.Atk"] = other_numbers[4].text
    pokemon["Sp. Def"] = other_numbers[5].text
    pokemon["Speed"] = other_numbers[6].text
    
    for t in type_list:
        pokemon[t] = 0
        
    for t in pokemon["Type"]:
        pokemon[t] = 1  
    
    poke_list.append(pokemon)
    
df = pandas.DataFrame(poke_list)
df.drop_duplicates(subset='Id', inplace=True)


# Fetch legendary pokemons
soup_leg = get_soup(legendary_url)
leg_table = soup_leg.find_all("table", {"class": "wikitable"})
leg_list = []
for table in leg_table:
    trs = table.find_all("tr")
    count = 0
    for tr in trs:
        if count == 0:
            count += 1
            continue
        leg_list.append(tr.find_all("td")[1].text.replace("\n", "").title())
df_leg = pandas.DataFrame(leg_list)


#Add a column with '1' if the pokemon is legendary, else 0
if "Legendary" in df:
    del df["Legendary"]
df["Legendary"] = df.Name.isin(df_leg[0]).apply(lambda row: 1 if row==True else 0)


# Time for some training
X_train, X_test, y_train, y_test = train_test_split(df.loc[: , : "Fairy"], df.loc[:, "Legendary"], test_size=0.33)

clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
# "Total" and "HP" seems to have to much weight, and the types effect seems minimal, skip those values
clf.fit(X_train.loc[:, "Attack":"Speed"], y_train)

preditions = clf.predict(X_test.loc[:, "Attack":"Speed"])
preditions_2 = clf.predict_proba(X_test.loc[:, "Attack":"Speed"])

print(matthews_corrcoef(y_test, preditions))

df_results = pandas.DataFrame()
df_results['POKWMAAAN'] = X_test['Name']
df_results['Predict'] = preditions
df_results['Predict2'] = preditions_2[:, 1]
df_results['Actual'] = y_test

importances = clf.feature_importances_