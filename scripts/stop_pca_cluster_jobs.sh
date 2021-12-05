#!/bin/bash

for M in 15 50 500
do
	for COMP in 50 100
	do
		kubectl delete job rna-seq-pca-cluster-comp-${COMP}-min-cluster-size-${M}
	done
done
