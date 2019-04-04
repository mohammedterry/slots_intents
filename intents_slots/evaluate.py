import json,pickle
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from model import BiLSTM_CRF
from utils import intent_entities

def evaluate(model,model_info,filepath,include_intents=True,include_entities=True):
  expected,predicted = [],[]
  with open(filepath) as f:
    for line in f.readlines():
      sentence,entitiesintent = line.strip().split('\t')
      entitiesintent = [entity.split('-')[-1].lower() for entity in entitiesintent.split()[1:]]
      entities,intent = entitiesintent[:-1],entitiesintent[-1:]
      pr_intent,pr_wordentities = intent_entities(' '.join(sentence.split()[1:-1]),model,model_info)
      if include_entities:
        pr_entities = [entity.split('-')[-1] if entity else 'o' for _,entity in pr_wordentities] 
        expected.extend(entities)
        predicted.extend(pr_entities)
      if include_intents and '#' not in intent[0]:
        expected.extend(intent)
        predicted.extend(pr_intent)
  labels = list(set(predicted+expected))
  score(expected,predicted,labels)

def score(y,y_hat,labels):
  print(classification_report(y,y_hat))
  cm = confusion_matrix(y,y_hat)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  cmat = pd.DataFrame(cm)
  cmat.columns = labels
  cmat.set_index([pd.Index(labels, '')],inplace=True)
  sns.heatmap(cmat,cmap="YlGnBu") #annot=True
  plt.title("Confusion Matrix")

def multiline_plot(data,keys,title,x_label,y_label): 
  sns.set_style("whitegrid")  

  for key in keys:
    plt.plot(data.index.values, key, data=data,
             marker = 'o', markersize = 2,linewidth = 2,
             markerfacecolor = ('blue','red')['val_' in key], 
             color=('skyblue','pink')['val_' in key], 
    )
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend()


#see test & validation loss (or accuracy) for pre-trained model over 50 epochs of training
with open('results/training_results.json') as f:
  datapoints = json.loads(f.read())
datapoints_df = pd.DataFrame(datapoints)

plt.subplot(321)
multiline_plot(data=datapoints_df,keys=['intent_slot_crf_loss','val_intent_slot_crf_loss'],title="Entity (slot) Loss",x_label= "epochs",y_label= "loss")
plt.subplot(322)
multiline_plot(data=datapoints_df,keys=['intent_slot_crf_accuracy','val_intent_slot_crf_accuracy'],title="Entity (slot) Classification",x_label= "epochs", y_label= "accuracy")
plt.subplot(323)
multiline_plot(data=datapoints_df,keys=['intent_classifier_output_loss','val_intent_classifier_output_loss'],title="Intent Loss",x_label= "epochs",y_label= "loss")
plt.subplot(324)
multiline_plot(data=datapoints_df,keys=['intent_classifier_output_categorical_accuracy','val_intent_classifier_output_categorical_accuracy'],title="Intent Classification",x_label= "epochs",y_label= "accuracy")
plt.subplot(325)
multiline_plot(data=datapoints_df,keys=['loss','val_loss'],title="Loss", x_label= "epochs", y_label= "loss")

#evauate entities & intent using Precision, Recall, F1 Scores, Confusion Matrix 
loaded_model = BiLSTM_CRF()
loaded_model.load("../pretrained_models/model.h5")
with open("../pretrained_models/dataset_info",'rb') as f:
  loaded_info = pickle.load(f)

plt.subplot(326)
evaluate(loaded_model, loaded_info, '../data/atis-test.csv', include_entities=False)




#show graphs together
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.show()

