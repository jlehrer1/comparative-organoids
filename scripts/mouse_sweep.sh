ITER=0 
for LR in 0.001 0.01 0.015 0.09 0.1 0.3 0.4 0.5 0.7 0.8
do 
    export LR=${LR} NUM=${ITER} && envsubst < yaml/models/retina_model.yaml | kubectl create -f - 
    ((ITER++))
done