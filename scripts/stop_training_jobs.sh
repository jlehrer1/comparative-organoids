for LAYERS in 4 10 20 40 60 100 200
do 
    for WIDTH in 512 1024 2048
    do 
        kubectl delete job rna-seq-model-weighting-width-${WIDTH}-layers-${LAYERS}
    done
done