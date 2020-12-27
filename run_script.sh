#!/bin/sh

date > log.txt
python3 ./helpers/check_req.py >> run_window_log.txt
if [ $? -eq 1 ]; then
    echo "Missing requirements"
    exit
fi


if [ "$1" != "" ]; then
    if [ "$2" != "" ]; then
        sh run.sh $1 $2| tee  run_window_log.txt
    else
        sh run.sh $1 syllable | tee  run_window_log.txt
    fi
else
    sh run.sh window syllable| tee  run_window_log.txt
fi


