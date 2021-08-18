# TRAVEL

## Introduction

This is an implementation of TRAVEL, a graph-based GNN framework proposed in the following paper:

[TRAVEL: Traffic Accident Prediction using Graph Neural Networks].
Baixiang Huang, Bryan Hooi.

Please cite our paper if you use the datasets or code in this repo.

## Demo

See the notebook `travel_accident_occurrence.ipynb` for example usages of our model.

## Models

The models are implemented in `travel_accident_occurrence.ipynb`.

## Datasets

Datasets for four major US cities are available in the directory `dataset_travel`.

The US-Accident dataset is available at https://smoosavi.org/datasets/us_accidents. 

The preprocessed TRAVEL datasets include:
- `x`, the node features,
- `y`, the labels for the traffic accident occurrence prediction task,
- `severity_labels`, the labels for the traffic accident severity prediction task,
- `edge_attr`, the edge features,
- `edge_attr_dir`, the directional edge features,
- `edge_attr_ang`, the angular edge features,
- `coords`, the node coordinates,
- `edge_index`, the graph indices.
