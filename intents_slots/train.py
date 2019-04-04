import pickle
import numpy as np
from tensorflow.python.keras.utils import to_categorical

from model import BiLSTM_CRF
from utils import ATIS, one_hot, wordvectors, embeddingmatrix

#Load in and format the training & test data
intent_dataset = ATIS(train_file='../data/atis-train.csv',test_file='../data/atis-test.csv',sentence_length=50,word_length=12)

#Convert all entity-slot labels into vectors (bows)
train_y = to_categorical(np.array(intent_dataset.data["train"]["entities"]), 1 + len(intent_dataset.info["entity_vocab"]))
test_y = to_categorical(np.array(intent_dataset.data["test"]["entities"]), 1 + len(intent_dataset.info["entity_vocab"]))

#Convert all intent labels into vectors (one-hot)
train_i = one_hot(intent_dataset.data["train"]["intents"], len(intent_dataset.info["intent_vocab"]))
test_i = one_hot(intent_dataset.data["test"]["intents"], len(intent_dataset.info["intent_vocab"]))

#Convert all words into vectors (poincare)
vec_dimensions = 100
wvectors = wordvectors('../word_vectors/poincare.txt')
embedding_matrix = embeddingmatrix(wvectors, vec_dimensions, intent_dataset.info["word_vocab"])

#Configure the Machine Learning Model
machinelearning_model = BiLSTM_CRF()
machinelearning_model.build(intent_dataset.info["word_length"], 1+len(intent_dataset.info["entity_vocab"]), len(intent_dataset.info["intent_vocab"]), 2+len(intent_dataset.info["word_vocab"]),2+len(intent_dataset.info["char_vocab"]),word_emb_dims=vec_dimensions,tagger_lstm_dims=100, dropout=0.2)
machinelearning_model.load_embedding_weights(embedding_matrix)

#Train the Machine Learning Model for 50 epochs
train_inputs = [intent_dataset.data["train"]["words"], intent_dataset.data["train"]["chars"]]
test_inputs = [intent_dataset.data["test"]["words"], intent_dataset.data["test"]["chars"]]
train_outputs = [train_i, train_y]
test_outputs = [test_i, test_y]
machinelearning_model.fit(train_inputs, train_outputs, batch_size = 32, epochs = 50, validation = (test_inputs, test_outputs))

#Save the trained Model and Vocab Mappings
machinelearning_model.save("model.h5")
with open("dataset_info", 'wb') as f:
  pickle.dump(intent_dataset.info, f)