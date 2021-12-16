CONTAINER = jmlehrer/cell-exploration

exec:
	docker exec -it $(CONTAINER) /bin/bash

build:
	docker build -t $(CONTAINER) .

push:
	docker push $(CONTAINER)

run:
	docker run -it $(CONTAINER) /bin/bash

go:
	make build && make push

train:
	./scripts/stop_training_jobs.sh; ./scripts/start_training_jobs.sh
