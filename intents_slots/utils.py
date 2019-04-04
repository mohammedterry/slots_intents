import numpy as np
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer 
wn_lemmatiser = WordNetLemmatizer()

def tokenise(sentence):
  tokens = ''.join([char if ord('a') <= ord(char.lower()) <= ord('z') or char == "'" or char.isdigit() else ' ' for char in f'{sentence} '.replace(':','').replace("`","'").replace('pm ',' pm ')])
  ts = []
  for token in tokens.split():
    if "am " in f'{token} ' and len(token) > 2 and token[-3].isdigit(): #avoid splitting words like ham, spam, sam, etc
      ts.extend([token[:-2],"am"])
    else:
      ts.append(token)
  return ts

def normalise(sentence): 
  return ["*" * len(token) if token.isdigit() else wn_lemmatiser.lemmatize(token.lower(),'v') for token in tokenise(sentence)]  

def one_hot(matrix, n_classes):  
  vector = np.zeros((np.array(matrix).shape[0], n_classes))
  for i,vs in enumerate(matrix):
    vector[i][vs] = 1.
  return vector

def wordvectors(file_path):
  vectors = {}
  with open(file_path) as f:
    for line in f.readlines():
      line = line.split()
      word,vector = line[0],np.array(line[1:],dtype='float32')
      vectors[word] = vector
  return vectors

def embeddingmatrix(wordvectors,vector_length,vocabulary):
  matrix = np.zeros((2+len(vocabulary),vector_length))
  for word,idx in vocabulary.items():
    if word.lower() in wordvectors:
      matrix[idx] = wordvectors[word.lower()]
  return matrix

def clip_pad(xs,length,pad=[0]):
  xs = [x[:length] for x in xs]  #clips it if too long
  xs = [x + pad * (length - len(x)) for x in xs] #pads it if too short
  return xs

def preprocess(sentence, model_info):
  ts = [model_info["word_vocab"][token] if token in model_info["word_vocab"] else 1 for token in normalise(sentence)]
  ts = clip_pad([ts],model_info["sentence_length"])
  cs = [[model_info["char_vocab"][char] if char in model_info["char_vocab"] else 1 for char in token] for token in tokenise(sentence)]
  cs = clip_pad([clip_pad(cs,model_info["word_length"])],model_info["sentence_length"],pad=[[0]*model_info["word_length"]])
  return [ts,cs]

def postprocess(model_output, model_info):
  ii,ee = model_output
  intents = [model_info["intent_idx"][i] for i in ii.argmax(1)]
  entities = [model_info["entity_idx"][e] if e in model_info["entity_idx"] else None for e in ee.argmax(2)[0]]
  return intents,entities

def intent_entities(sentence,model,model_info): #wrapper: -> input string -> output intents & entities
  intent,entities = postprocess(model.predict(preprocess(sentence,model_info)),model_info) 
  return intent, list(zip(tokenise(sentence),entities))

class ATIS():
    def __init__(self, train_file, test_file, sentence_length=30, word_length=12):
      self.info = {
        "sentence_length":sentence_length,
        "word_length":word_length,
      }
      self._load(self._parse(train_file), self._parse(test_file))

    def _parse(self,file_path):
      with open(file_path, encoding='utf-8', errors='ignore') as f:
        parsed_lines = []
        for line in f.readlines():
          words,entities_intent = line.strip().split('\t')
          entities,intent = entities_intent.split()[1:-1],entities_intent.split()[-1]
          if "#" not in intent:
            parsed_lines.append((tokenise(words)[1:-1],normalise(words)[1:-1],entities,intent))
        return parsed_lines

    def _w_vectors_vocab(self, sentences, start=0):
      idx_vocab = dict(enumerate({word.lower() for sentence in sentences for word in sentence},start))
      vocab_idx = {v:k for k,v in idx_vocab.items()}
      vectors = [[vocab_idx[word.lower()] for word in sentence] for sentence in sentences]
      return vectors, vocab_idx, idx_vocab  
    
    def _c_vectors_vocab(self, sentences, start = 0):
      idx_vocab = dict(enumerate({char for sentence in sentences for word in sentence for char in word},start))
      vocab_idx = {v:k for k,v in idx_vocab.items()}
      vectors = [[[vocab_idx[char] for char in word] for word in sentence] for sentence in sentences]
      return vectors, vocab_idx, idx_vocab

    def _load(self, train_set, test_set):
      #join train + test data together and split into words,entities,intents
      words, tokens, entities, intents = list(zip(*train_set + test_set))        

      #build vocab for each and encode each as categorical vectors
      word_vectors, self.info["word_vocab"], self.info["word_idx"] = self._w_vectors_vocab(tokens, start=2)
      word_vectors = clip_pad(word_vectors, self.info["sentence_length"])
      
      entities_vectors, self.info["entity_vocab"], self.info["entity_idx"] = self._w_vectors_vocab(entities, start=1)
      entities_vectors = clip_pad(entities_vectors, self.info["sentence_length"])

      intents_vectors, self.info["intent_vocab"], self.info["intent_idx"] = self._w_vectors_vocab([intents])
      intents_vectors = intents_vectors[0]

      chars_vectors, self.info["char_vocab"], self.info["char_idx"] = self._c_vectors_vocab(words, start=2)
      chars_vectors = [clip_pad(word, self.info["word_length"]) for word in chars_vectors]     
      chars_vectors = clip_pad(chars_vectors,self.info["sentence_length"],pad=[[0]*self.info["word_length"]]) 
                                     
      #separate train/test data
      train_size, test_size = len(train_set), len(test_set)
      self.data = {
        "train": {
            "words":word_vectors[:train_size],
            "chars":chars_vectors[:train_size],
            "intents":intents_vectors[:train_size],
            "entities":entities_vectors[:train_size],
        },
        "test" : {
            "words":word_vectors[-test_size:],
            "chars":chars_vectors[-test_size:],
            "intents":intents_vectors[-test_size:],
            "entities":entities_vectors[-test_size:],
        },
      }