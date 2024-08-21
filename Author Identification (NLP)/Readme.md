# Spooky Author Identification

This project involves identifying the authorship of texts written by three famous authors: Edgar Allan Poe, H.P. Lovecraft, and Mary Shelley. The model utilizes GloVe embeddings for word vectorization and an LSTM-based neural network for classification.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [References](#references)

## Project Overview

The goal of this project is to classify text excerpts from three different authors using Natural Language Processing (NLP) techniques. The main components of this project include:

1. Data Preprocessing: Cleaning the text data, tokenization, and transforming the text into word vectors using GloVe.
2. Model Building: Developing an LSTM-based neural network to classify the texts.
3. Model Evaluation: Assessing the model's performance using accuracy and log loss metrics.

## Dataset

The dataset consists of text excerpts attributed to Edgar Allan Poe, H.P. Lovecraft, and Mary Shelley. Each text is labeled with the author's name, which serves as the target variable for classification.

## Dependencies

To run the code in this notebook, you'll need the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- TensorFlow/Keras
- nltk

## Data Preprocessing

Key preprocessing steps include:

- Text cleaning (removing punctuation, converting to lowercase, etc.)
- Tokenization of text data.
- Converting text to word vectors using pre-trained GloVe embeddings.
- Padding sequences to ensure uniform input size for the LSTM model.

## Modeling

The primary model used in this project is an LSTM-based neural network. The model architecture includes:

- An embedding layer using GloVe vectors.
- LSTM layers to capture sequential dependencies in the text.
- Dense layers for classification.

Hyperparameter tuning and model optimization are performed to achieve the best performance.

## Evaluation

The model's performance is evaluated based on:

- Accuracy
- Log Loss

Confusion matrices and classification reports are used to provide detailed performance insights.

## Conclusion

The results of the model are discussed, along with potential areas for improvement. The model demonstrates the capability to effectively classify text based on authorship with reasonable accuracy.

## How to Run

To run the project locally:

1. Clone this repository: `git clone https://github.com/yourusername/spooky-author-identification.git`
2. Navigate to the project directory: `cd spooky-author-identification`
3. Install the necessary dependencies: `pip install -r requirements.txt`
4. Run the notebook: `jupyter notebook Spooky_Author_Identification_(GloVe_+_LSTM).ipynb`

## References

- GloVe Embeddings: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
- TensorFlow/Keras Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)

