#!/bin/bash

for N in 15 50 500
do
	for COMP in 50 100
	do
	for M in 50 100 250
		do
			kubectl delete job rna-seq-cluster-n-${N}-comp-${COMP}-min-cluster-size-${M}
		done
	done
done


