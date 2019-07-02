# Tiresias

A computational framework for disease gene prioritization.

## Version

Tiresias is still under development. The present software is available in beta.

## Getting started

This software has been tested on Ubuntu 18.04. It is meant to be run on Linux systems.

### Prerequisites
* [Conda](https://conda.io/en/latest/)
* [Docker](https://www.docker.com/)

### Installing

1. Clone the repository and get into the Git directory.

````bash
git clone git@github.com:RausellLab/Tiresias.git && cd Tiresias
````

2. Setup a Conda environment and activate it.

````bash
conda env create -f environment.yml && conda activate tiresias
````

3. Pull node2vec Docker image.

````bash
make node2vec_image
````

You can now check that the framework works properly by launching the data pipeline with dummy input data:

````bash
make pipeline
````

You can then browse the results with [MLFlow](https://mlflow.org/) UI. Launch:

````bash
mlflow ui
````

Then, go to [http://localhost:5000/](http://localhost:5000/) in your web browser.

Once you're done, clean up the intermediary and output files created by the pipeline:

````bash
rm -rf artifacts/ tmp/ mlflow/
````

## Usage

### Configuration

1. Datasets

Input datasets paths must be entered in `config.yml`.

2. Parameters

Set up the parameters used for featurization and for running the models by modifying the JSON files located in `parameters/`.

* `parameters/features.json`: contains the parameters used to run random walks and to create node embeddings.
* `parameters/models_validation.json`: contains model parameters used during the **validation** stage.
* `parameters/models_test.json`: contains model parameters used during the **test** stage.

3. System and models

System resource configuration and enabling/disabling the use of models is done by modifying `config.yml`.

### Run pipeline

Before running any pipeline step, the Conda environment must be activated with:

````bash
conda activate tiresias
````

The pipeline can be run from start to finish with a single command:

````bash
make pipeline
````

It is also possible to run the pipeline steps individually.

1. Data preprocessing
````bash
make data
````

2. Random walks
````bash
make random_walks
````

3. Node embeddings
````bash
make embeddings
````

4. Validation
````bash
make validation
````

5. Test
````bash
make test
````

### Visualize results with MLFlow

Model run results are recorded using [MLFlow](https://mlflow.org/). Results are stored in the `mlflow/` directory.

You can browse results with MLFlow UI. To do so, launch MLFlow UI.
````bash
mlflow ui
````

Then, go to [http://localhost:5000/](http://localhost:5000/) in your web browser.

### Intermediary files

Intermediary files generated when running pipeline steps and their associated metadata are stored in `artifacts/`.

Note that the pipeline scripts automatically pick-up the files present in `artifacts/` as inputs for downstream pipeline steps.
This allow for more flexibility when running pipeline steps separately. However, you may want not to repeat some experiments or start with
completely different data or parameters. In this case, move or delete the existing `artifacts/` directory, then re-run
the pipeline from start with the new inputs.

## Built with

* [PyTorch](https://pytorch.org/)
* [Deep Graph Library](https://www.dgl.ai/)
* [SNAP](http://snap.stanford.edu/)
* [Gensim](https://radimrehurek.com/gensim/)
* [MLFlow](https://mlflow.org/)
* [Ray](https://github.com/ray-project/ray)

## References

````
Mordelet, F. and Vert, J.-P. (2011). ProDiGe: Prioritization Of Disease Genes with multitask machine learning from positive and unlabeled examples. BMC Bioinformatics, 12(1):389.

Mordelet, F. and Vert, J.-P. (2014). A bagging SVM to learn from positive and unlabeled examples. Pattern Recognition Letters, 37:201–209.

Lovász, László, et al. Random walks on graphs: A survey. Combinatorics, Paul erdos is eighty, 1993, vol. 2, no 1, p. 1-46.

Zhou, D., Bousquet, O., Lal, T. N., Weston, J., and Schölkopf, B. (2004). Learning with local and global consistency. In Advances in Neural Information Processing Systems 16, pages 321–328. MIT Press

Li, Y. and Li, J. (2012). Disease gene identification by random walk on multigraphs merging heterogeneous genomic and phenotype data. BMC Genomics, 13(7):S27.

Valdeolivas, A., Tichit, L., Navarro, C., Perrin, S., Odelin, G., Levy, N., ... & Baudot, A. (2018). Random walk with restart on multiplex and heterogeneous biological networks. Bioinformatics, 35(3), 497-505.

Grover, A. and Leskovec, J. (2016). node2vec: Scalable Feature Learning for Networks. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining - KDD ’16, pages 855–864, San Francisco, California, USA. ACM Press.

T. N. Kipf and M. Welling, “Semi-Supervised Classification with Graph Convolutional Networks,” arXiv:1609.02907 [cs, stat], Sep. 2016.

M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, and M. Welling, “Modeling Relational Data with Graph Convolutional Networks,” arXiv:1703.06103 [cs, stat], Mar. 2017.
````

## Licence

Tiresias uses Apache License 2.0.

## Contact

Please address comments and questions about Tiresias to thibaud.martinez@gmail.com, stefani.dritsa@institutimagine.org, chloe-agathe.azencott@mines-paristech.fr and antonio.rausell@inserm.fr.
