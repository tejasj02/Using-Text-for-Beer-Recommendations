import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import pathlib

def predict_beer(custom_review):
    ''' Predict the beer name based on a text review '''

    model_path = pathlib.Path("saved_model").resolve().as_posix()

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

    label_encoder = joblib.load(pathlib.Path(model_path) / "label_encoder.pkl")

    # Tokenize 
    inputs = tokenizer(
        custom_review,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=128
    )

    # Send inputs to GPU if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Run model inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Decode predicted label back to beer name
    predicted_beer = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_beer


