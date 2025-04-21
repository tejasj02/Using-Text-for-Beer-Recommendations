import ast
import pandas as pd

def save_data():
    ''' This function reads the beeradvocate.json file and saves the data to a CSV file. '''

    data = []
    with open('beeradvocate.json', 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(ast.literal_eval(line.strip()))
            except Exception as e:
                print("Skipping line due to error:", e)

    df = pd.DataFrame(data)
    df = df[['beer/beerId', 'beer/name', 'review/text']]
    df = df.dropna()  # Remove rows with missing text
    df.head()

    top_beers = df['beer/beerId'].value_counts().head(20).index
    df_small = df[df['beer/beerId'].isin(top_beers)]
    df_small.to_csv("beer_reviews_20.csv", index=False)
    return
