# TRAVEL

## Introduction

This is an implementation of Traﬃc Accident Vulnerability Estimation via Linkage (TRAVEL), a graph-based GNN framework proposed in the following paper:

[Traffic Accident Prediction using Graph Neural Networks: New Datasets and the TRAVEL Model].
Baixiang Huang, Bryan Hooi.

Please cite our paper if you use the datasets in this repo.


## Datasets

Datasets for 20 US cities are available in the directory `dataset_travel`.

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

You can use the code below to load them.
```python
class TRAVELDataset(InMemoryDataset):
    r"""The TRAVEL datasets from the
    `". Traffic Accident Prediction using Graph Neural Networks: New Datasets and the TRAVEL Model" 
    <https://link>`_ paper.
    Nodes represent intersections or dead-end nodes and edges represent roads.
    Datasets include 'houston', 'charlotte', 'dallas', 'austin', 'miami', 'raleigh', 
    'atlanta', 'baton_rouge', 'nashville', 'orlando', 'oklahoma_city', 'sacramento', 
    'phoenix', 'minneapolis', 'san_diego', 'seattle', 'richmond', 'san_antonio', 
    'jacksonville', 'saint_paul'

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    
    url = 'https://github.com/baixianghuang/travel/raw/main/dataset/{}.npz'
    
    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url.format(self.name), self.raw_dir)

    def process(self):
        data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}Full()'


TRAVELDataset('file_path', city_name)
```

In addition, raw features are also provided. 
```python
# Raw node features include 'y', 'x', 'street_count', 'highway', 'ref', 'geometry', 'accident_cnt', 'severity', 'start_time', 'end_time'
with open('your_file_path/'+city_name+'_raw_node_attr.pkl', 'rb') as fp:
    gdf_nodes_raw = pickle.load(fp)

# Raw edge features include 'osmid', 'oneway', 'name', 'highway', 'length', 'ref', 'lanes', 'geometry', 'bridge', 'maxspeed', 'access', 'tunnel', 'junction'
with open('your_file_path/'+city_name+'_raw_edge_attr.pkl', 'rb') as fp:
    gdf_edges_raw = pickle.load(fp)
```
