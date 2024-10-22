# TAP: Traffic Accident Prediction


## Introduction
<!-- This is an implementation of Traffic Accident Vulnerability Estimation via Linkage (TRAVEL), a graph neural network framework proposed in the following paper: -->
This repo contains the Traffic Accident Prediction (TAP) datasets proposed in the following paper:

TAP: A Comprehensive Data Repository for Traffic Accident Prediction in Road Networks. *ACM SIGSPATIAL* (2023) [link](https://arxiv.org/pdf/2304.08640)

```
@article{huang2023tap,
  title={TAP: A Comprehensive Data Repository for Traffic Accident Prediction in Road Networks},
  author={Huang, Baixiang and Hooi, Bryan and Shu, Kai},
  journal={arXiv preprint arXiv:2304.08640},
  year={2023}
}
```


## Datasets
The Traffic Accident Prediction (TAP) data repository offers extensive coverage for 1,000 US cities (TAP-city) and 49 states (TAP-state), providing real-world road structure data that can be easily used for graph-based machine learning methods such as Graph Neural Networks. Additionally, it features multi-dimensional geospatial attributes, including angular and directional features, that are useful for analyzing transportation networks. The TAP repository has the potential to benefit the research community in various applications, including traffic crash prediction, road safety analysis, and traffic crash mitigation. The datasets can be accessed in the TAP-city and TAP-state directories.

For example, this repository can aid in traffic accident occurrence prediction and accident severity prediction. Binary labels are used to indicate whether a node contains at least one accident for the occurrence prediction task, while severity is represented by a number between 0 and 7 for the severity prediction task. A severity level of 0 denotes no accident, and 1 to 7 represents increasingly significant impacts on traffic.

The table below shows the features included in our datasets. 

 Graph features             | Description               
----------------------------|------------------------------------------------
 highway                    | The type of a road (tertiary, motorway, etc.). 
 length                     | The length of a road.                          
 bridge                     | Indicates whether a road represents a bridge.  
 lanes                      | The number of lanes of a road.                 
 oneway                     | Indicates whether a road is a one-way street.  
 maxspeed                   | The maximum legal speed limit of a road.       
 access                     | Describes restrictions on the use of a road.   
 tunnel                     | Indicates whether a road runs in a tunnel.     
 junction                   | Describes the junction type of a road.                    
 street\_count              | The number of roads connected to a node.   

Each dataset contains a single, directed road garph. Each road network is described by an instance of 'torch_geometric.data.Data', which include following attributes:
- `x`: Node feature matrix with shape `[num_nodes, num_node_features]`
- `y`: Node-level labels for the traffic accident occurrence prediction task
- `severity_labels`: Node-level labels for the traffic accident severity prediction task
- `edge_attr`: Edge feature matrix with shape `[num_edges, num_edge_features]`
- `edge_attr_dir`: Directional edge feature matrix with shape `[num_edges, 2]`
- `edge_attr_ang`: Angular edge feature matrix with shape `[num_edges, 3]`
- `coords`: Node coordinates
- `crash_time`: Start and end timestamps for traffic accidents with shape `[2, num_nodes]`
- `edge_index`: Graph indices in COO format (coordinate format) with shape `[2, num_edges]` and type torch.long


Please refer to `code/detailed_node_and_edge_features.ipynb` for the detailed feature names corresponding to each feature in `x` and `edge_attr`.


List of 1,000 cities sorted by their total counts of traffic accident occurrences.
```python
with open('util/cities_sorted_by_accident.pkl', 'rb') as fp:
    cities_sorted_by_accident = pickle.load(fp)
```


### Enivorment
Run the following commands to create an environment and install all the required packages:
```bash
conda env create -f environment.yml
```
In addition, you can access and run the code from Google Colab: [occurrence_task](https://colab.research.google.com/drive/13VpVN3fKupaYsieDRYBSxqkUdeGyEK-t?usp=sharing) and [severity_task](https://colab.research.google.com/drive/1AjO2BKqWdbLgXO4ObwOa0AF_fRYlfSFj?usp=sharing).
<!-- conda config --prepend channels conda-forge
conda create -n tap --strict-channel-priority osmnx
conda activate tap
pip install -r requirements.txt -->


You can use the code below to load the datasets.
```python
import torch
import numpy as np
import os.path as osp
from typing import Union, Tuple, Callable, Optional
from torch_geometric.data import Data, DataLoader, InMemoryDataset, download_url
          
class TRAVELDataset(InMemoryDataset):
    r"""
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
    
    url = 'https://github.com/baixianghuang/travel/raw/main/TAP-city/{}.npz'
    # url = 'https://github.com/baixianghuang/travel/raw/main/TAP-state/{}.npz'
    
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

def read_npz(path):
    with np.load(path, allow_pickle=True) as f:
        return parse_npz(f)


def parse_npz(f):
    crash_time = f['crash_time']
    x = torch.from_numpy(f['x']).to(torch.float)
    coords = torch.from_numpy(f['coordinates']).to(torch.float)
    edge_attr = torch.from_numpy(f['edge_attr']).to(torch.float)
    cnt_labels = torch.from_numpy(f['cnt_labels']).to(torch.long)
    occur_labels = torch.from_numpy(f['occur_labels']).to(torch.long)
    edge_attr_dir = torch.from_numpy(f['edge_attr_dir']).to(torch.float)
    edge_attr_ang = torch.from_numpy(f['edge_attr_ang']).to(torch.float)
    severity_labels = torch.from_numpy(f['severity_8labels']).to(torch.long)
    edge_index = torch.from_numpy(f['edge_index']).to(torch.long).t().contiguous()
    return Data(x=x, y=occur_labels, severity_labels=severity_labels, edge_index=edge_index, 
                edge_attr=edge_attr, edge_attr_dir=edge_attr_dir, edge_attr_ang=edge_attr_ang, 
                coords=coords, cnt_labels=cnt_labels, crash_time=crash_time)


dataset = TRAVELDataset('file_path', 'Dallas_TX')
data = dataset[0]
print(f'Number of graphs: {len(dataset)}')
print(f'Number of node features: {dataset.num_features}')
print(f'Number of edge features: {dataset.num_edge_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
print(f'Contains self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
```

State-level datasets with size over 100 MB are stored in [Google Drive](https://drive.google.com/drive/folders/1tgxbEgnuFAAi1VMg4fTn-m1LA4Zbex4m?usp=sharing)

Please note that we do not have ownership of the data and therefore not in a position to provide a license or control its use. However, we kindly request that the data only be used for research purposes.


<!-- ## License
TAP is distributed under the terms of the Apache License (Version 2.0). -->