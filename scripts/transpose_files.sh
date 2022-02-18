#!/bin/bash

COUNT=0
for FILE in 'primary_bhaduri.tsv' 'allen_cortex.tsv' 'allen_m1_region.tsv' 'whole_brain_bhaduri.tsv'
do
    export FILE=${FILE} COUNT=${COUNT} && envsubst < yaml/transpose_individual.yaml | kubectl create -f - 
    (( COUNT ++ ))
done 