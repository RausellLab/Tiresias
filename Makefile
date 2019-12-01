SHELL := /bin/bash

data-multi-layer:
	python -m src.pipeline.data.multi_layer_network.make_adjacency_matrices
	python -m src.pipeline.data.multi_layer_network.make_edge_lists

data-merged-layer:
	python -m src.pipeline.data.merged_layer_network.make_adjacency_matrix
	python -m src.pipeline.data.merged_layer_network.make_edge_list

data:
	$(MAKE) data-multi-layer data-merged-layer

random-walks:
	python -m src.pipeline.features.random_walks

embeddings:
	python -m src.pipeline.features.embeddings

validation:
	python -m src.pipeline.validation.main

test:
	python -m src.pipeline.test.main

predict:
	python -m src.pipeline.predict.main

pipeline:
	$(MAKE) data random-walks embeddings validation test predict

CLEAN_INTERM_MSG := "WARNING: This will erase intermediary directories (artifacts/ and tmp/). Continue? (Y/N): "
clean-interm:
	@read -p $(CLEAN_INTERM_MSG) confirm && [[ $$confirm == [yY] || $$confirm == [yY][eE][sS] ]] || exit 1
	rm -rf artifacts/ tmp/

CLEAN_ALL_MSG := "WARNING: This will erase ALL generated files. Continue? (Y/N): "
clean-all:
	@read -p $(CLEAN_ALL_MSG) confirm && [[ $$confirm == [yY] || $$confirm == [yY][eE][sS] ]] || exit 1
	rm -rf artifacts/ tmp/ mlruns/

node2vec-image:
	python -m src.init.pull_node2vec

.PHONY: data-multi-layer data-merged-layer data random-walks embeddings validation test predict pipeline node2vec-image
.DEFAULT_GOAL := pipeline
