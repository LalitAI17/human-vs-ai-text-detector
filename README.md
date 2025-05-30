# Human vs AI Text Detector

A machine learning application that distinguishes between human-written and AI-generated text using various natural language processing techniques and models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Web Application](#running-the-web-application)
  - [Using the Interface](#using-the-interface)
  - [Using the Models Programmatically](#using-the-models-programmatically)
- [Technical Details](#technical-details)
  - [Models](#models)
  - [Project Structure](#project-structure)
- [Development](#development)
- [Performance](#performance)
- [Requirements](#requirements)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

## Overview

This project implements a text classification system capable of detecting whether a given text sample was written by a human or generated by an AI system. It leverages multiple machine learning approaches, including Sentence Transformers and fine-tuned RoBERTa models, to achieve accurate classification results.

## Features

- Web-based user interface built with Streamlit
- Multiple classification models:
  - Transformer-based embeddings with SVM classifier
  - Fine-tuned RoBERTa model
- Automatic model downloading from cloud storage
- Easy-to-use prediction interface

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/human-vs-ai-text-detector.git
   cd human-vs-ai-text-detector
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the necessary models:
   The first time you run the application, it will automatically download the required models.

## Usage

### Running the Web Application

Start the Streamlit web application:

```bash
streamlit run web_app.py
```

Visit the URL provided in the terminal (typically http://localhost:8501) to access the web interface.

### Using the Interface

1. Enter or paste the text you want to analyze into the text area
2. Select the model you want to use for classification
3. Click the "Predict" button
4. View the prediction result (AI Generated or Human Written)

### Using the Models Programmatically

You can also use the classification functions directly in your Python code:

```python
from text_preprocessing_and_model import sentence_embedding_based_classifier, roberta_model_based_classifier

# Using the sentence embedding model
result1 = sentence_embedding_based_classifier("Your text to classify goes here")
print(result1)  # Will print "AI Generated" or "Human Written"

# Using the RoBERTa model
result2 = roberta_model_based_classifier("Your text to classify goes here")
print(result2)  # Will print "AI Generated" or "Human Written"
```

## Technical Details

### Models

1. **Sentence Embedding Model**

   - Uses the all-MiniLM-L6-v2 model from Sentence Transformers
   - Text is encoded into dense vector representations
   - An SVM classifier makes the final prediction

2. **RoBERTa Model**
   - Fine-tuned RoBERTa model with a custom classification head
   - Handles text tokenization and feature extraction
   - Trained to directly predict text origin

### Project Structure

```
human-vs-ai-text-detector/
├── web_app.py               # Streamlit web application
├── text_preprocessing_and_model.py  # Model implementation
├── requirements.txt         # Python dependencies
├── models/                  # Directory for storing model files
│   └── roberta_model/       # RoBERTa model files
├── notebooks/               # Jupyter notebooks for development
│   ├── Sentence_Transformer_+_Traditional_Classifier.ipynb
│   ├── Tf-idf_+_Traditional_Classifier.ipynb
│   ├── Context_Vector_+_Traditional_Classifier.ipynb
│   └── Finetuning_of_Roberta_model
└── results_and_visualizations/  # Performance metrics and visualizations
```

## Development

The development process and model training are documented in the Jupyter notebooks located in the `notebooks/` directory. These notebooks include:

- Data preprocessing techniques
- Feature extraction methods
- Model training and evaluation
- Performance comparisons

## Performance

The models have been evaluated on a balanced dataset containing both human-written and AI-generated texts. Performance metrics include accuracy, precision, recall, and F1-score.

For detailed performance analysis, refer to the visualizations and results in the `results_and_visualizations/` directory.

## Requirements

- Python 3.8 or higher
- PyTorch
- Transformers
- Sentence-Transformers
- Streamlit
- Scikit-learn
- CUDA-compatible GPU (optional, for faster inference)

See `requirements.txt` for the complete list of dependencies.

## Limitations

- Performance may vary depending on the source of AI-generated text
- Shorter texts may be harder to classify accurately
- The models work best with English language text

## Future Work

- Integration of additional language models
- Support for multi-language classification
- Improvements to handle adversarial examples
- Enhanced explanations of classification decisions

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for the transformer models
- [Sentence-Transformers](https://www.sbert.net/) for the embedding models
- [Streamlit](https://streamlit.io/) for the web application framework
