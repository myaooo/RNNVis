#!/bin/bash
project_dir="$1"
server_ip="$2"
ssh -t $server_ip 'cd $project_dir; ./test.sh'