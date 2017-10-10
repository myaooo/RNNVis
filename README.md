# RNNVis

A visualization tool for understanding and debugging RNNs.

**Note**: This is an underdeveloped project.

## Goals

The major goal of this project is to explore possible ways to help better unerstanding of RNN models (Vanilla RNN, LSTM, GRU, etc.)
and help practitioners to debug their model and data, and help reasearchers improve model architecture and performances.

## Setup

1. Install TensorFlow r0.12.1 (gpu is also supported)

2. To install all the dependency packages, under the project dir, run:
 
    `pip install -r requirements.txt` 

3. Example datasets are already in the cached_data dir.
 
   To set up the mongodb, you must first have mongodb installed on your system and have a `mongod` instance running. [More details](https://docs.mongodb.com/manual/administration/install-community/)
   
   Then, at the project root dir, run
   
   `python -m rnnvis.main seeddb`

## TODO

- [ ] Remove the dependency of Mongo for easier extension.

- [ ] Migration to TensorFlow 1.0 and Vuex.

## Usage

### Training an RNN model

1. Run tests on PTB datasets to see whether the code runs normally: 

    `python -m tests.test_language_model --config_path=./config/model/lstm.yml --data_path=./cached_data/simple-examples/data`

2. For a well performed model, and use pre-defined procedures, run:

    `python -m tests.test_procedures --config_path=./config/model/lstm-large3.yml`

    you can modify the model file under the `/config` directory and customize your model and training parameters.


### Visualizing the hidden states of a model

1. Since the system is built on top of the hidden states of RNN. You need to first record the hidden states of the model by running it on a dataset (the hidden states record will be stored in mongodb for latter use). To do so, you can run:

    `python -m tests.test_eval_record --config_path=./config/model/lstm-large3.yml`

2. Then, to run the visualization server, first modify the `./config/models.yml` to config which models you want to load. Then run:

    `python -m rnnvis.main server` 

    to host a server for the visualization. Also note that it will take some time for the visualization to pop up (depend on the size of your model and the dataset) the first time you attempt to run the visualization.


