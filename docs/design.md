# Program Design

The design of the classes and logic are similar to the design of TFLearn.

## Class design

### DNN

Base class of RNNs and CNNs, which nests Trainer and Summarizer, with APIs including:

* infer (predict)
* evaluate (accuracy, etc.)
* training
* save & load
* set & get weights,

### RNN(DNN)

DNN with recurrent layers, manage recurrent layers (cells) within the model

### Trainer

A model trainer, wrapping up TF training ops