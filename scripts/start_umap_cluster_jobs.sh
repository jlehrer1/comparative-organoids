#!/bin/bash

for N in 15 50 500
do
	for COMP in 50 100
	do
	for M in 50 100 250
		do
			export N=${N} COMP=${COMP} M=${M} && envsubst < yaml/cluster_umap.yaml | kubectl create -f -
		done
	done
done
