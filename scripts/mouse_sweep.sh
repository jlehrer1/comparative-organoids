ITER=0 
for LR in 0.01 0.09 0.1 0.3
do 
    export LR=${LR} NUM=${ITER} && envsubst < yaml/models/retina_model.yaml | kubectl create -f - 
    ((ITER++))
done