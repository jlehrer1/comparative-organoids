#!/bin/bash

for N in 15 50 100 500
do
	for COMP in 2 3 50 100
	do
		export N=${N} COMP=${COMP} && envsubst < yaml/cluster.yaml | kubectl create -f -
	done
done
