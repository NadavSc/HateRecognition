import os
import re
import requests
import pandas as pd
import wikipedia as wiki

from bs4 import BeautifulSoup
from data_parser import wikis


def wiki_txt_clean(text, include_n=True):
    text = re.sub(r'==.*?==+', '', text)
    if include_n:
        return text
    return text.replace('\n', '')


# Scrap text only (not tables)
def wiki_txt_scrap(wiki_name):
    wiki_txt = wiki.page(wiki_name)
    return wiki_txt_clean(wiki_txt.content)


def is_col_equal(df1, df2):
    if len(df1.columns) == len(df2.columns):
        return (df1.columns == df2.columns).all()
    return False


def dataframes_combine(dfs):
    dfs_generator = (df for df in dfs)
    df_main = next(dfs_generator)

    for df in dfs_generator:
        if is_col_equal(df_main, df):
            df_main = pd.concat([df_main, df])
        else:
            break
    return df_main


def read_html(response):
    soup = BeautifulSoup(response.text, 'html.parser')
    indiatable = soup.select('table', {'class': "wikitable"})
    return pd.read_html(str(indiatable))


def wiki_tables_scrap(url):
    return read_html(requests.get(url))


wiki_url = "https://en.wikipedia.org/wiki/List_of_ethnic_slurs"
df = wiki_tables_scrap(url=wiki_url)
df = dataframes_combine(df)
txt = wiki_txt_scrap('LGBT slang')
