# TRAVEL

## Introduction

This is an implementation of TRAVEL, a graph-based GNN framework proposed in the following paper:

[Traffic Accident Prediction using Graph Neural Networks: New Datasets and the TRAVEL Model].
Baixiang Huang, Bryan Hooi.

Please cite our paper if you use the datasets or code in this repo.

## Models

The models are implemented in `accident_occurrence.ipynb` and `accident_severity.ipynb`.

## Datasets

Datasets for four major US cities (Houston, Charlotte, Dallas, and Austin) are available in the directory `dataset_travel`.

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

In addition, raw OpenStreetMap features are also provided. You can use the code below to load them.
```python
# Raw node features include 'y', 'x', 'highway', 'street_count', 'ref', 'geometry', 'accident_cnt', 'Severity'
with open('your_file_path/'+city_name+'_raw_node_attr.pkl', 'rb') as fp:
    gdf_nodes_raw = pickle.load(fp)

# Raw edge features include 'osmid', 'oneway', 'name', 'highway', 'length', 'ref', 'lanes', 'geometry', 'bridge', 'maxspeed', 'access', 'tunnel', 'junction'
with open('your_file_path/'+city_name+'_raw_edge_attr.pkl', 'rb') as fp:
    gdf_edges_raw = pickle.load(fp)
```
