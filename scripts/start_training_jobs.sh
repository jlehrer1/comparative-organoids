for LAYERS in 5 15 60 
do 
    for WIDTH in 512 1024 2048
    do 
        export LAYERS=${LAYERS} WIDTH=${WIDTH} && envsubst < yaml/model.yaml | kubectl create -f - 
    done
done
    