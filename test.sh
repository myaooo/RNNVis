#!/bin/bash
source ~/tensorflow-gpu/bin/activate
nohup `python -m py.test_lstm` > test_out.log &