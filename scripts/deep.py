import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import Trainer, TrainingArguments
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from scripts.deep_data import BeerReviewDataset

def get_data():
    ''' Load your dataset (already filtered to top 20 beers) '''
    ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
    data_path = ROOT_DIR / "data" / "beer_reviews_20.csv"
    df = pd.read_csv(data_path)

    df = df[['review/text', 'beer/name']].dropna()

    # Encode beer names as labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['beer/name'])

    # Train-val-test split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=42)
    return train_df, val_df, test_df, label_encoder

def set_model():
    ''' Set up bert model and freeze/unfreeze layers '''
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=20)

    # Freeze all
    for param in model.distilbert.parameters():
        param.requires_grad = False

    # Unfreeze last 2 layers
    for i in [4, 5]:
        for param in model.distilbert.transformer.layer[i].parameters():
            param.requires_grad = True

    for param in model.distilbert.embeddings.parameters():
        param.requires_grad = True
    
    return model, tokenizer

def set_training(model, tokenizer, train_df, val_df):
    ''' Set up training params '''
    train_dataset = BeerReviewDataset(train_df, tokenizer)
    val_dataset = BeerReviewDataset(val_df, tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        fp16=True,  # Speeds up training on your GPU
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    return trainer

def train_model(trainer):
    ''' Train the model '''
    trainer.train()
    trainer.evaluate()
    return trainer

def evaluate_model(trainer, tokenizer, label_encoder ,test_df):
    ''' Evaluate the model with classifcation report'''
    test_dataset = BeerReviewDataset(test_df, tokenizer)
    predictions = trainer.predict(test_dataset)
    pred_labels = predictions.predictions.argmax(axis=1)

    from sklearn.metrics import classification_report
    return classification_report(test_df['label'], pred_labels, target_names=label_encoder.classes_)

def deep_main():
    train_df, val_df, test_df, label_encoder = get_data()
    model, tokenizer = set_model()
    trainer = set_training(model, tokenizer, train_df, val_df)
    trained_model = train_model(trainer)
    report = evaluate_model(trained_model, tokenizer, label_encoder, test_df)
    return report