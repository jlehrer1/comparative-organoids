#!/bin/bash

for FILE in 'organoid' 'primary'
do
	for N in 2 3
	do
		export N=${N} FILE=${FILE} && envsubst < yaml/pca.yaml | kubectl create -f - 
	done
done
