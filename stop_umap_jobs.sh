#!/bin/bash

for N in 10 15 
do
	for COMP in 2 3 50 100
	do
		kubectl delete job rna-seq-reduction-${N}-neighbors-${COMP}-components
	done
done
