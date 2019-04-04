# Intent and Slot-filling on the ATIS dataset task.

## Model architecture
For this task, a model was trained to jointly predict a sentence's Intent and Slots (entities).  Each word is embedded (using pre-defined word vectors) to capture the word's meaning while a character-level bidirection Long Short-Term Memory(LSTM) Network encodes the word's letters to capture its lexical structure.  

![](intents_slots/architecture.png)

The word vector and outputs of the character-level bi-LSTM are then fed into the word-level bi-LSTM which predicts the Intent.  The second layer feeds into a Conditional Random Fields (CRF) layer to predict the individual slots (entities).

The model is stored in ```intents_slots/model.py``` 

## Word Vectors
![](word_vectors/poincare.jpg) 
The newly released Poincare word embeddings (100 dimensional) were used as they have been reported to better encode the hierarchical relationships inherent between words
![](word_vectors/hierarchical_word_relations.png) 
you can find the word vectors used in ```word_vectors/poincare.txt```

## Demo
you can run a demo of the pre-trained model by running  ```intents_slots/demo.py```
![](intents_slots/results/demo.png)
![](intents_slots/results/demo2.png)

## Training
The model was trained for 50 epochs and stored in the pretrained_models directory 
- ```pretrained_models/dataset_info``` contains all the vocabularies used by the model (character, word, intent, entity) and their mappings to numbers for encoding/decoding
- ```pretrained_models/model.h5``` are the weights to the model

The model's loss during training over the epochs are shown below:
![](intents_slots/results/intent_training_loss.png)
![](intents_slots/results/entity_training_loss.png)

The model's accuracy at predicting intents and entities (slots) over time are shown below:
![](intents_slots/results/intent_training_accuracy.png)
![](intents_slots/results/entity_training_accuracy.png)

You can retrain the model by running ```intents_slots/train.py```

## Tests
Joint:
![](intents_slots/results/f1_entityintent.png)

Intents only:
![](intents_slots/results/f1_intents.png)

Entites only:
![](intents_slots/results/f1_entities.png)

to above results can be obtained by running ```intents_slots/evaluate.py```

## Improvements:
![](intents_slots/results/normalisation.png)

To improve the robustness of the model to out of vocab words, the training data was lemmatised prior to training and the model was retrained.  Numbers were also masked using a placeholder (e.g. *) to avoid out-of-vocab times appearing (e.g. 9:30 may appear in training but not 9:29).  

![](intents_slots/results/before_after.png)

The results were slightly improved given the above tweaks.  Precision, Recall and F1 scores improved across the board (for both intents, entities).
![](intents_slots/results/training_normalisation.png)
![](intents_slots/results/f1_intents_normalisation.png)
![](intents_slots/results/f1_entities_normalisation.png)
![](intents_slots/results/f1_entityintent_normalisation.png)

Perfect scores were achieved using the validation set!?? ```data/atis-2-dev.csv```
![](intents_slots/results/f1_validation_intents.png)
![](intents_slots/results/f1_validation_all.png)

## Future Improvements (TO DO):
- Balance out training data (its clear that the intent ATIS_FLIGHT dominates the training set) 

![](intents_slots/results/intent_support_graph.png)

(and 'O' dominates the entity tags) 
![](intents_slots/results/entities_support_withO_graph.png)

(or if we discount this as an entity tag - then "to/fromloc.city_name" tags)
![](intents_slots/results/entities_support_graph.png)

- e.g. this can be achieved by subsampling or artificially perterbing data to generate more samples (e.g. increase training instance by sliding each sentences one,two,three,etc places)

- Investigate the Intents & Entities which are scoring relatively low F1 scores

e.g. (intents such as ATIS_DAY_NAME, ATIS_MEAL, ATIS_FLIGHT_TIME, etc)

![](intents_slots/results/intent_scores_graph.png)

e.g. (entities such as compartment, booking_class, meal_code, etc)

![](intents_slots/results/entity_scores_graph.png)

- Preprocess intent labels with #?
- Embed unknown words too (if possible) rather than giving them <UNK> (1)
- convert word numbers (e.g. "one") into digits
- improve slot extraction using additional pre-trained Named Entity Recognition (NER)s from various libraries

![](intents_slots/results/pretrained_ners.png)
![](intents_slots/results/pretrained_ners_type.png)