# ArtBench-10 trial on Spark

[![Readme-JA](https://img.shields.io/badge/README-Japanese-red.svg)](README.ja.md)

This is an expandable trial based on the `ArtBench-10` dataset([32x32 CIFAR-python](https://artbench.eecs.berkeley.edu/files/artbench-10-python.tar.gz)), using the `Spark` distributed computing framework. In the trial, a convolutional generative adversarial network(see `CGAN.py`) was trained. And notebook `test_after_training_CGAN.ipynb` tested the generation effect.

## Original paper:
Liao, P., Li, X., Liu, X., & Keutzer, K. (2022). The ArtBench Dataset: Benchmarking Generative Models with Artworks. [ArXiv, abs/2206.11404](https://arxiv.org/abs/2206.11404).

## Original repository: 
[https://github.com/liaopeiyuan/artbench](https://github.com/liaopeiyuan/artbench)

## CGAN.py usage:

```bash
spark-submit CGAN.py \
# custom source assignment, for example,
--master <your Spark standalone url> \
--total-executor-cores 4 \
--executor-cores 1 \
--executor-memory 2G \
--driver-memory 4G \
```

## test_after_training_CGAN.ipynb usage:
After training the network, run cells in notebook. The following case figure:
<img src='./generated_images.png' width='60%'>

##
I would like to express my sincere gratitude to the original authors.