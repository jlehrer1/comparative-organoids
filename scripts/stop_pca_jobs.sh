#!/bin/bash

for FILE in 'organoid' 'primary'
do
	for N in 2 3
	do
		kubectl delete job rna-seq-pca-${N}-components-${FILE}
	done
done
