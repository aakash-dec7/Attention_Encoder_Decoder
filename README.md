# Attention Encoder Decoder

## Overview
This project implements a simple encoder-decoder sequence-to-sequence (seq2seq) model with Bahdanau Attention. The model is designed for neural machine translation tasks, such as translating English sentences into Spanish. It leverages PyTorch to build and train the architecture.

## Features
* Implements **Bahdanau Attention** mechanism to enhance the decoder's ability to focus on relevant parts of the input sequence.
* Implements seq2seq architecture with LSTM layers for both Encoder and Decoder.
* Supports tokenization, padding, and batch processing.
* Ability to save and load training checkpoints.
* Configurable hyperparameters for flexibility in model design.
* Uses PyTorch for building and training the model.
* Implemented BLEU score evaluation to assess the performance of the model.

## Dataset
The dataset used for training consists of paired sentences in English and Spanish.
Data Preprocessing includes:
* Adding special tokens (`<sos>` and `<eos>`) to target sentences.
* Tokenization and padding of sentences to a maximum length.
* Conversion of tokenized data into PyTorch tensors for training.

The dataset should be stored in a CSV file named eng_spn.csv, with two columns:
* English words/sentences
* Spanish words/sentences

## Architecture
The model consists of the following components:

**1. Encoder**
* Embedding Layer: Maps input tokens to dense vectors.
* LSTM Layer: Encodes input sequences into hidden and cell states.
* Dropout Layer: Prevents overfitting by randomly deactivating neurons during training.

**2. Bahdanau Attention**
* Computes a context vector based on the decoder's current hidden state and encoder outputs.
* Employs linear layers and a softmax operation for attention weight calculation.

**3. Decoder**
* Embedding layer for target sequences.
* Multi-layer LSTM with dropout.
* Attention mechanism for context computation.
* Fully connected layer for generating token probabilities.

**4. Seq2Seq**
* Combines the Encoder and Decoder for end-to-end sequence prediction.

## Hyperparameters
Adjust these to customise your training:
* Sequence Length: `MAX_LENGTH = 25`
* Embedding Dimensions: `EMBEDDING_DIM = 128`
* Hidden Dimensions: `HIDDEN_DIM = 256`
* Layers: `NUM_LAYERS = 2`
* Dropout: `DROPOUT = 0.3`
* Batch Size: `BATCH_SIZE = 64`
* Epochs: `NUM_EPOCHS = 50`
* Learning Rate: `LEARNING_RATE = 0.001`

## Contributions
Feel free to fork the repository and submit a pull request for improvements.

## License
This project is licensed under the MIT License.
