#!/bin/bash

for N in 15 50 100 500
do
	for COMP in 2 3 50 100
	do
		kubectl delete job rna-seq-cluster-n-${N}-comp-${COMP}
	done
done
