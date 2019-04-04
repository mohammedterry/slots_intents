import pickle
import pandas as pd
from model import BiLSTM_CRF
from utils import intent_entities
  
#load in pretrained model & corresponding vocab mappings
loaded_model = BiLSTM_CRF()
loaded_model.load("../pretrained_models/model.h5")
with open("../pretrained_models/dataset_info",'rb') as f:
  loaded_info = pickle.load(f)

while True:
  intent,entities = intent_entities(input("> "),loaded_model,loaded_info)

  es = {"WORDS":[],"SLOTS (ENTITIES)":[]}
  for word,entity in entities:
    es["WORDS"].append(word)
    es["SLOTS (ENTITIES)"].append(entity)
  print(f"INTENT:\t{intent[0].upper()}\n\n{pd.DataFrame(es)}\n")