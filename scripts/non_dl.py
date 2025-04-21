import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report

# Load data
def data_load():
    ''' Load your dataset (already filtered to top 20 beers) '''
    ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
    data_path = ROOT_DIR / "data" / "beer_reviews_20.csv"
    df = pd.read_csv(data_path)
    df = df[['review/text', 'beer/name']].dropna()

    # Train/val/test split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['beer/name'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['beer/name'], random_state=42)
    beer_texts_base = train_df.groupby('beer/name')['review/text'].apply(lambda x: ' '.join(x)).reset_index()
    return beer_texts_base, val_df, test_df

def tune(beer_texts_base, val_df):
    ''' Train a TF-IDF model to predict beer names from reviews with validation for hyperparameter tuning '''

    # Try different max_features values
    max_features_options = [1000, 3000, 5000, 10000]
    best_acc = 0
    best_max_feat = None

    for max_feat in max_features_options:
        vectorizer = TfidfVectorizer(max_features=max_feat)
        tfidf_matrix = vectorizer.fit_transform(beer_texts_base['review/text'])

        def predict_beer(review):
            vec = vectorizer.transform([review])
            sims = cosine_similarity(vec, tfidf_matrix)
            return beer_texts_base['beer/name'].iloc[sims[0].argmax()]

        val_df['predicted'] = val_df['review/text'].apply(predict_beer)
        acc = accuracy_score(val_df['beer/name'], val_df['predicted'])

        if acc > best_acc:
            best_acc = acc
            best_max_feat = max_feat

        return best_max_feat

def train(beer_texts_base, best_max_feat):
    ''' Train the final model with the best hyperparameters '''
    final_vectorizer = TfidfVectorizer(max_features=best_max_feat)
    final_matrix = final_vectorizer.fit_transform(beer_texts_base['review/text'])
    return final_vectorizer, final_matrix

def evaluate(test_df, vectorizer, matrix, beer_names):
    ''' Evaluate the model on the test set '''
    def predict(review):
        vec = vectorizer.transform([review])
        sims = cosine_similarity(vec, matrix)
        return beer_names.iloc[sims[0].argmax()]

    test_df['predicted'] = test_df['review/text'].apply(lambda x: predict(x))
    return classification_report(test_df['beer/name'], test_df['predicted'])

def non_dl_main():
    ''' Main function for the non-deep learning approach '''
    beer_texts_base, val_df, test_df = data_load()
    best_max_feat = tune(beer_texts_base, val_df)
    vectorizer, matrix = train(beer_texts_base, best_max_feat)
    report = evaluate(test_df, vectorizer, matrix, beer_texts_base['beer/name'])
    return report