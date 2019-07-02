pipeline: data random_walks embeddings validation test

data: data_merged_layer data_multi_layer

data_multi_layer:
	python -m src.pipeline.data.multi_layer_network.make_adjacency_matrices
	python -m src.pipeline.data.multi_layer_network.make_edge_lists

data_merged_layer:
	python -m src.pipeline.data.merged_layer_network.make_adjacency_matrix
	python -m src.pipeline.data.merged_layer_network.make_edge_list

random_walks:
	python -m src.pipeline.features.random_walks

embeddings:
	python -m src.pipeline.features.embeddings

validation:
	python -m src.pipeline.validation.main

test:
	python -m src.pipeline.test.main

node2vec_image:
	python -m src.init.pull_node2vec

.PHONY: pipeline data data_multi_layer data_merged_layer random_walks embeddings validation test
