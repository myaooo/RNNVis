#!/bin/bash

# Activating TF virtual environment
source ~/tensorflow-gpu/bin/activate

# Run test_language_model
nohup python -m py.test_language_model --config_path=./config/lstm.yml --data_path=./cached_data/simple-examples/data > test_lm.log &

# Run test_procedures
nohup python -m py.test_procedures --config_path=./config/lstm-large.yml --data_path=./cached_data/simple-examples/data > test_pc.log &

# Run test_generator
nohup python -m py.test_procedures --config_path=./config/lstm-large.yml --data_path=./cached_data/simple-examples/data > test_gn.log &


