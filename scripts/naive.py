import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def naive_results():
    ''' Naive classifier: always predicts the most common beer '''

    ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
    data_path = ROOT_DIR / "data" / "beer_reviews_20.csv"
    df = pd.read_csv(data_path)
    
    df = df[['review/text', 'beer/name']].dropna()
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['beer/name'])
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=42)

    most_common_beer = train_df['beer/name'].mode()[0]

    # Predict that for every test row
    test_df['predicted'] = most_common_beer

    # Evaluate
    return classification_report(test_df['beer/name'], test_df['predicted'])