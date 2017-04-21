#!/bin/bash
# runs DiscriminitveAnalysis.py 100 times and outputs to a log file ShuffledRun.log

COUNTER=0;
while [ $COUNTER -lt 100 ]; do 
	./ClassScript.sh >> ShuffledRun.log; 
	echo '_________________________________________________________' >> ShuffledRun.log; 
	let COUNTER+=1; 
done
