#!/bin/bash

for FILE in 'organoid' 'primary'
do
	for N in 15 50 100 500
	do
		kubectl delete job rna-seq-pca-${N}-components-${FILE}
	done
done
