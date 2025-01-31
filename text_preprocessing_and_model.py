from sentence_transformers import SentenceTransformer
import joblib
import pandas as pd
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
import torch
import torch.nn as nn
import json
import gdown
import os


def download_model(url, output):
    gdown.download(url, output, quiet=False, fuzzy=True)



# Mapping of prediction labels to human-readable text
label_mapping = {0: 'AI Generated', 1: 'Human Written'}


def load_model(model_path: str) -> object:
    """
    Load a trained model from a specified file path.

    Args:
        model_path (str): Path to the serialized model file.

    Returns:
        object: The loaded model object.
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as error:
        raise ValueError(f"Failed to load model from file: {error}") 
    

def sentence_embedding_based_classifier(input_text : str):
    input_text_list = [input_text]

    if not isinstance(input_text_list, list) or not all(isinstance(i, str) for i in input_text_list):
        raise ValueError("input_text must be a list of strings.")

    # Load the sentence embedding model
    sentence_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate sentence embeddings
    sentence_embedding = sentence_embedding_model.encode(input_text_list)
    sentence_embedding = pd.DataFrame(
        sentence_embedding, 
        columns=[f'Feature_{i+1}' for i in range(sentence_embedding.shape[1])]
    )

    model_url = r"https://drive.google.com/file/d/1JjV_F6E5CuxK8mhlUi_bK9x1JVU2AYlJ/view?usp=sharing"

    # check folder exits or not - sentence_embedding_model
    if not os.path.exists(r"models\sentence_embedding_model"):
        os.makedirs(r"models\sentence_embedding_model", exist_ok=True)
    
    model_path = r"models\sentence_embedding_model\pipeline_svm.joblib"

    if not os.path.exists(model_path):
        download_model(model_url, model_path)

    model = load_model(model_path)

    # Perform prediction
    output = model.predict(sentence_embedding)
    predicted_label = label_mapping.get(output[0], "Unknown")
    return predicted_label    

class FineTunedRobertaClassifier(nn.Module):
    def __init__(self, roberta_model, roberta_tokenizer, num_class, device):
        super().__init__()
        self.model = roberta_model
        self.tokenizer = roberta_tokenizer
        self.device = device

        for param in self.model.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(in_features=768, out_features=768),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(768, num_class)
        )

    def forward(self, input_text):
        text_embedding = self.tokenizer(
            input_text,
            max_length = 512,
            truncation = True,
            padding = 'max_length',
            return_tensors = 'pt'
        ).to(self.device)
        last_hidden_state = self.model(**text_embedding).last_hidden_state
        logits = self.classifier(last_hidden_state[:, 0, :])
        return logits
    

def roberta_model_based_classifier(input_text : str) -> torch.Tensor:

    roberta_tokenizer = RobertaTokenizer.from_pretrained(r"models/roberta_model/roberta_tokenizer")
    with open(r"models/roberta_model/roberta_model_config.json", 'r') as file:
        config_file = json.load(file)
    config = RobertaConfig.from_dict(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FineTunedRobertaClassifier(RobertaModel(config), roberta_tokenizer, num_class = 2, device = device) 

    model_url = r"https://drive.google.com/file/d/1_8aPwcP6mTkS0gLktVsAL1oJm1tH6Vq7/view?usp=sharing"
    model_path = r"models/roberta_model/best_model.pth"

    if not os.path.exists(model_path):
        download_model(model_url, model_path)

    best_model_dict = torch.load(model_path, map_location=device, weights_only=True)

    model.load_state_dict(best_model_dict["model_state_dict"])
    model.eval()
    with torch.no_grad():
        prediction = model(input_text)
        predicted_label = label_mapping.get(prediction.argmax().item(), "Unknown")
    return predicted_label


if __name__ == "__main__":
    pred = roberta_model_based_classifier("This is an example sentence for testing")
    print(pred)
    
    output = sentence_embedding_based_classifier(input_text="This is an example sentence for testing")
    print(output)
    
    
