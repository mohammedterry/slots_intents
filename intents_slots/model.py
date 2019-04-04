import pickle, tempfile
from tensorflow.keras import Model
from tensorflow.train import AdamOptimizer
from tensorflow.keras.layers import Input,Embedding,Dropout,Dense,TimeDistributed,Bidirectional,LSTM,concatenate
from crf_layer import CRF

class BiLSTM_CRF():
  
  def build(self,word_length,num_labels,num_intent_labels,word_vocab_size,char_vocab_size,word_emb_dims=100,char_emb_dims=30,char_lstm_dims=30,tagger_lstm_dims=100,dropout=0.2):

    self.word_length = word_length
    self.num_labels = num_labels
    self.num_intent_labels = num_intent_labels
    self.word_vocab_size = word_vocab_size
    self.char_vocab_size = char_vocab_size
    
    words_input = Input(shape=(None,), name='words_input')
    embedding_layer = Embedding(word_vocab_size,word_emb_dims, name='word_embedding')
    word_embeddings = embedding_layer(words_input)
    word_embeddings = Dropout(dropout)(word_embeddings)

    word_chars_input = Input(shape=(None, word_length),name='word_chars_input')
    char_embedding_layer = Embedding(char_vocab_size, char_emb_dims,input_length=word_length,name='char_embedding')
    char_embeddings = char_embedding_layer(word_chars_input)
    char_embeddings = TimeDistributed(Bidirectional(LSTM(char_lstm_dims)))(char_embeddings)
    char_embeddings = Dropout(dropout)(char_embeddings)

    # first BiLSTM layer (used for intent classification)
    first_bilstm_layer = Bidirectional(LSTM(tagger_lstm_dims, return_sequences=True, return_state=True))
    first_lstm_out = first_bilstm_layer(word_embeddings)

    lstm_y_sequence = first_lstm_out[:1][0]  # save y states of the LSTM layer
    states = first_lstm_out[1:]
    hf, _, hb, _ = states  # extract last hidden states
    h_state = concatenate([hf, hb], axis=-1)
    intents = Dense(num_intent_labels, activation='softmax', name='intent_classifier_output')(h_state)
    # create the 2nd feature vectors
    combined_features = concatenate([lstm_y_sequence, char_embeddings],axis=-1)

    # 2nd BiLSTM layer (used for entity/slots classification)
    second_bilstm_layer = Bidirectional(LSTM(tagger_lstm_dims,return_sequences=True))(combined_features)
    second_bilstm_layer = Dropout(dropout)(second_bilstm_layer)
    bilstm_out = Dense(num_labels)(second_bilstm_layer)

    # feed BiLSTM vectors into CRF
    crf = CRF(num_labels, name='intent_slot_crf')
    entities = crf(bilstm_out)

    model = Model(inputs=[words_input, word_chars_input],outputs=[intents, entities])

    loss_f = {'intent_classifier_output': 'categorical_crossentropy','intent_slot_crf': crf.loss}
    metrics = {'intent_classifier_output': 'categorical_accuracy', 'intent_slot_crf': crf.viterbi_accuracy}
    model.compile(loss=loss_f, optimizer= AdamOptimizer(), metrics=metrics)
    self.model = model
    
  def load_embedding_weights(self, weights):
    self.model.get_layer(name='word_embedding').set_weights([weights])
    
  def fit(self, x, y, epochs=1, batch_size=1, callbacks=None, validation=None):
    self.model.fit(x, y, epochs=epochs, batch_size=batch_size, shuffle=True,validation_data=validation,callbacks=callbacks)

  def predict(self, x, batch_size=1):
    return self.model.predict(x, batch_size=batch_size)

  def save(self, filepath):
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as f:
      self.model.save_weights(f.name)
      data = {'model_weights': f.read(), 'model_topology': {k:v for k,v in self.__dict__.items() if k != "model"}}
    with open(filepath, 'wb') as f:
      pickle.dump(data, f)
        
  def load(self,filepath):
    with open(filepath, 'rb') as f:
      model_data = pickle.load(f)
    self.build(**model_data['model_topology'])
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as f:
      f.write(model_data['model_weights'])
      f.flush()
      self.model.load_weights(f.name)
