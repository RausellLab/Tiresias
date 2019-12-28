SHELL := /bin/bash

.PHONY: data-multi-layer
data-multi-layer:	## Preprocess the input data for the multi-layer network.
	python -m src.pipeline.data.multi_layer_network.make_adjacency_matrices
	python -m src.pipeline.data.multi_layer_network.make_edge_lists

.PHONY: data-merged-layer
data-merged-layer:	## Merge network layers and preprocess the resulting data.
	python -m src.pipeline.data.merged_layer_network.make_adjacency_matrix
	python -m src.pipeline.data.merged_layer_network.make_edge_list

.PHONY: data
data:	## Preprocess the input data.
	$(MAKE) data-multi-layer data-merged-layer

.PHONY: random-walks
random-walks:	## Simulate random walks on the networks.
	python -m src.pipeline.features.random_walks

.PHONY: embeddings
embeddings:	## Generate vector embeddings of the nodes.
	python -m src.pipeline.features.embeddings

.PHONY: validation
validation:	## Run the validation step of the pipeline.
	python -m src.pipeline.validation.main
	$(MAKE) validation-best-runs-report

.PHONY: test
test:	## Run the test step of the pipeline.
	python -m src.pipeline.test.main
	$(MAKE) test-best-runs-report

.PHONY: predict
predict:	## Run the prediction step of the pipeline.
	python -m src.pipeline.predict.main
	$(MAKE) prediction-report

.PHONY: validation-best-runs-report
validation-best-runs-report:	## Generate a report of the best runs for the validation step.
	python -m src.reports.best_runs_by_model validation

.PHONY: test-best-runs-report
test-best-runs-report:	## Generate a report of the best runs for the test step.
	python -m src.reports.best_runs_by_model test

.PHONY: prediction-report
prediction-report:	## Generate a report of the predictions. The report is available in reports/.
	python -m src.reports.predictions

.PHONY: reports
reports:	## Generate all reports.
	$(MAKE) validation-best-runs-report
	$(MAKE) test-best-runs-report
	$(MAKE) prediction-report

.PHONY: pipeline
pipeline:	## Run the whole pipeline.
	$(MAKE) data random-walks embeddings validation test predict

.PHONY: clean-interm
CLEAN_INTERM_MSG := "WARNING: This will erase intermediary directories (artifacts/ and tmp/). Continue? (Y/N): "
clean-interm:	## Erase intermediary directories artifacts/ and tmp/.
	@read -p $(CLEAN_INTERM_MSG) confirm && [[ $$confirm == [yY] || $$confirm == [yY][eE][sS] ]] || exit 1
	rm -rf artifacts/ tmp/

.PHONY: clean-all
CLEAN_ALL_MSG := "WARNING: This will erase ALL generated files. Continue? (Y/N): "
clean-all:	## Erase ALL generated files.
	@read -p $(CLEAN_ALL_MSG) confirm && [[ $$confirm == [yY] || $$confirm == [yY][eE][sS] ]] || exit 1
	rm -rf artifacts/ tmp/ mlruns/ reports/

.PHONY: node2vec-image
node2vec-image:	## Pull node2vec image from the Docker registry.
	python -m src.init.pull_node2vec

.PHONY: help
help:	## Display this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
