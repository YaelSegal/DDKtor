#!/bin/sh

date > log.txt
python3 ./helpers/check_req.py >> run_log.txt
if [ $? -eq 1 ]; then
    echo "Missing requirements"
    exit
fi


if [ "$1" != "" ]; then
    sh run.sh $1| tee  run_log.txt
else
    sh run.sh window | tee  run_log.txt
fi


