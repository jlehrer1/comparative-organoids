for LAYERS in 4 10 20 40 60 100 200
do 
    for WIDTH in 512 1024 2048
    do 
        export LAYERS=${LAYERS} WIDTH=${WIDTH} && envsubst < yaml/model.yaml | kubectl create -f - 
    done
done
    