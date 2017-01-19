#!/bin/bash
project_dir="$1"
server_ip="$2"
echo $project_dir
ssh $server_ip 'cd '${project_dir}' ;pwd; git pull; ./test.sh'