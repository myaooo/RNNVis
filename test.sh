#!/bin/bash
source ~/tensorflow-gpu/bin/activate
nohup `python -m py.test_language_model --config_path=./config/lstm.yml --data_path=./cached_data/simple-examples/data` > test_out.log &
