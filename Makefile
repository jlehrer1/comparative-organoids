CONTAINER = jmlehrer/cell-exploration

exec:
	docker exec -it $(CONTAINER) /bin/bash

build:
	docker build -t $(CONTAINER) .

push:
	docker push $(CONTAINER)

run:
	docker run -it $(CONTAINER) /bin/bash

all:
	kubectl create -f yaml/umap.yaml && kubectl create -f yaml/transpose.yaml 

stop:
	kubectl delete job rna-seq-generate-transpose; kubectl delete job rna-seq-umap; 
