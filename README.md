# RNNVis
A visualization tool for understanding and debugging RNNs.
**Note**: This is an underdeveloped project.
## Goals
The major goal of this project is to explore possible ways to help better unerstanding of RNN models (Vanilla RNN, LSTM, GRU, etc.)
and help practitioners to debug their model and data, and help reasearchers improve model architecture and performances.

## Setup

1. Install TensorFlow r0.12.1 (gpu is also supported)

2. to install all the dependency packages, under the project dir, run:
 
    `pip install -r requirements.txt` 

3. Data sets are already in the `data` dir.
 
   To set up the mongodb, you must first have mongodb installed on your system and have a `mongod` instance running.
   
   Then, at the project root dir, run
   
   `python -m rnnvis.main seeddb`

## Usage

1. Run tests on PTB datasets to see whether the code runs normally: 

    `python -m tests.test_language_model --config_path=./config/lstm.yml --data_path=./data/simple-examples/data`

2. For a well performed model, and use pre-defined procedures, run:

    `python -m tests.test_procedures --config_path=./config/lstm-large3.yml --data_path=./data/simple-examples/data`
