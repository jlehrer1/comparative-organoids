export TYPE=mouse && export NAME=mouse && envsubst < yaml/models/model.yaml | kubectl create -f - 
export TYPE=retina && export NAME=retina && envsubst < yaml/models/model.yaml | kubectl create -f -
export TYPE=dental && export NAME=dental && envsubst < yaml/models/model.yaml | kubectl create -f -