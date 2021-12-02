#!/bin/bash

for FILE in 'organoid' 'primary'
do
	for N in 15 50 100 500 5000
	do
		for COMP in 2 3 50 100
		do
			kubectl delete job rna-seq-reduction-${N}-neighbors-${COMP}-components-${FILE}
		done
	done
done
