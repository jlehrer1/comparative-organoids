#!/bin/bash

for M in 15 50 500
do
	for COMP in 50 100
	do
		export M=${M} COMP=${COMP} && envsubst < yaml/cluster_pca.yaml | kubectl create -f -
	done
done
