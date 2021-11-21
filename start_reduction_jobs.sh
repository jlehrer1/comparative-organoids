#!/bin/bash

for N in 10 15 50 500 5000
do
	for COMP in 2 3 50 100
	do
		export N=${N} COMP=${COMP} && envsubst < yaml/reduction.yaml | kubectl create -f -
	done
done
