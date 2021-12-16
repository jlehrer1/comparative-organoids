for LAYERS in 5 15 60 
do 
    for WIDTH in 512 1024 2048
    do 
        kubectl delete job rna-seq-model-weighting-width-${WIDTH}-layers-${LAYERS}
    done
done