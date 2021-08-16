# TRAVEL

## Introduction

This is an implementation of TRAVEL, a graph-based GNN framework proposed in the following paper:

[TRAVEL: Traffic Accident Prediction using Graph Neural Networks].
Baixiang Huang, Bryan Hooi.

Please cite the above paper if you use the datasets or code in this repo.

## Run the demo

We include the TRAVEL dataset in the directory `dataset_travel`, where the data structures needed are pickled.

You can refer to `travel_accident_occurrence.ipynb` for example usages of our model.

## Models

The models are implemented in `travel_accident_occurrence.ipynb`.

## Prepare the data

### Preprocessed datasets

Datasets for 4 major US cities are available in the directory `data`, in a preprocessed format.

The US-Accident dataset is available at https://smoosavi.org/datasets/us_accidents. We also provide a much more succinct version of the dataset that only contains necessary coordinates information in the directory `accident_coordinates`.

The preprocessed TRAVEL datasets include:
- `x`, the node features,
- `y`, the labels for the traffic accident occurrence prediction task,
- `labels_severity`, the labels for the traffic accident severity prediction task,
- `edge_attr`, the edge features,
- `edge_index`, the list of index tuples,
- `gdf_edges`, the OpenStreetMap edge dataframe,
- `gdf_nodes`, the OpenStreetMap node dataframe.

You can use the pickle package to load the objects `x`, `y`, `edge_attr`, `edge_index`, `gdf_edges`, `gdf_nodes`, `labels_severity`.

