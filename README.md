# Traffic Accident Prediction

## Introduction

<!-- This is an implementation of Traffic Accident Vulnerability Estimation via Linkage (TRAVEL), a graph neural network framework proposed in the following paper: -->
This repo contains the Traffic Accident Benchmark (TAB-1k) datasets proposed in the following paper:

Traffic Accident Prediction using Graph Neural Networks: New Datasets and the TRAVEL Model. Baixiang Huang, Bryan Hooi.

Please cite our paper if you use the datasets in this repo.

## Datasets

This is a new set of Traffic Accident Benchmark (TAB-1k) for one thousand US cities. We build the datasets based on real-world geospatial data from [OpenStreetMap](https://www.openstreetmap.org/) and the US-Accident datasets [(Moosavi et al., 2019)](https://arxiv.org/abs/1909.09638). Datasets are available in the directory `dataset`.

These benchmark datasets can be used for two kind of tasks: traffic accident occurrence prediction and accident severity prediction. For the accident occurrence prediction task, we use binary labels to indicate whether a node contains at least one accident. For the severity prediction task, severity is represented by a number between 0 and 7, where 0 denotes no accident, 1 indicates the most negligible impact on traffic, and 7 indicates a significant impact on traffic. All environmental features obtained from OpenStreetMap are preprocessed. 

The table below shows the OpenStreetMap features included in our datasets. Note that accident coordinates and timestamps are also included in the datasets.

 Graph feature              | Description               
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
- `edge_index`: Graph indices in COO format (coordinate format) with shape `[2, num_edges]` and type torch.long

List of 1k cities sorted by their total counts of traffic accident occurrences.
```python
with open('util/cities_sorted_by_accident.pkl', 'rb') as fp:
    cities_sorted_by_accident = pickle.load(fp)
```

The table below shows the statistics of top 50 cities.
| City                  | # nodes | # edges | Avg node  degree | % accident  nodes |
|-----------------------|:-------:|:-------:|:----------------:|:-----------------:|
| Los Angeles (CA)      |  49713  |  136742 |       2.75       |       12.46       |
| Houston (TX)          |  59694  |  149281 |        2.5       |        33.9       |
| Charlotte (NC)        |  29364  |  68403  |       2.33       |       31.15       |
| Dallas (TX)           |  36022  |  92117  |       2.56       |       30.39       |
| Austin (TX)           |  25549  |  63554  |       2.49       |       35.63       |
| Miami (FL)            |   8365  |  22537  |       2.69       |       11.01       |
| New York City (NY)    |  55361  |  140298 |       2.53       |       16.05       |
| Raleigh (NC)          |  15576  |  37236  |       2.39       |       33.65       |
| Nashville (TN)        |  21750  |  54292  |        2.5       |       31.47       |
| Atlanta (GA)          |  13210  |  34513  |       2.61       |       14.67       |
| Baton Rouge (LA)      |   8899  |  23771  |       2.67       |       44.96       |
| Orlando (FL)          |   7434  |  18140  |       2.44       |       29.28       |
| Oklahoma City (OK)    |  28194  |  73203  |        2.6       |       25.94       |
| Sacramento (CA)       |  13561  |  35250  |        2.6       |       11.25       |
| Phoenix (AZ)          |  47744  |  122346 |       2.56       |       11.79       |
| Minneapolis (MN)      |   7741  |  23543  |       3.04       |        9.88       |
| San Diego (CA)        |  26848  |  66116  |       2.46       |        7.68       |
| Seattle (WA)          |  19021  |  50227  |       2.64       |       26.56       |
| San Antonio (TX)      |  38480  |  98836  |       2.57       |       19.16       |
| Saint Paul (MN)       |   7009  |  20601  |       2.94       |        8.99       |
| Jacksonville (FL)     |  37514  |  91175  |       2.43       |        6.14       |
| Richmond (VA)         |   7867  |  20996  |       2.67       |       40.98       |
| Portland (OR)         |  20277  |  57202  |       2.82       |       18.63       |
| San Jose (CA)         |  20618  |  49366  |       2.39       |        8.27       |
| Indianapolis (IN)     |  35875  |  91825  |       2.56       |       15.03       |
| Greenville (SC)       |   3077  |   8110  |       2.64       |       19.63       |
| Columbia (SC)         |   4966  |  13638  |       2.75       |       12.34       |
| Chicago (IL)          |  28668  |  76242  |       2.66       |       15.71       |
| Denver (CO)           |  16736  |  48063  |       2.87       |       19.77       |
| Tucson (AZ)           |  19775  |  50127  |       2.53       |       15.55       |
| Omaha (NE)            |  15843  |  42957  |       2.71       |        25.3       |
| Tulsa (OK)            |  17689  |  46797  |       2.65       |       25.65       |
| Tampa (FL)            |  12399  |  35456  |       2.86       |         11        |
| Rochester (NY)        |   4802  |  13057  |       2.72       |       50.27       |
| Dayton (OH)           |   5296  |  14931  |       2.82       |       43.32       |
| Detroit (MI)          |  20848  |  59890  |       2.87       |        8.88       |
| Flint (MI)            |   4150  |  12487  |       3.01       |       47.88       |
| Oakland (CA)          |   8468  |  22294  |       2.63       |       12.82       |
| Riverside (CA)        |   7554  |  18681  |       2.47       |        7.64       |
| Fort Lauderdale (FL)  |   5109  |  13565  |       2.66       |        9.96       |
| Grand Rapids (MI)     |   5401  |  14986  |       2.77       |        45.2       |
| Louisville (KY)       |  21747  |  53637  |       2.47       |        8.53       |
| Long Beach (CA)       |   7168  |  19686  |       2.75       |       10.13       |
| Columbus (OH)         |  22232  |  58021  |       2.61       |        9.01       |
| Salt Lake City (UT)   |   5193  |  13821  |       2.66       |       12.11       |
| El Paso (TX)          |  21752  |  56794  |       2.61       |       15.87       |
| Anaheim (CA)          |   6629  |  15513  |       2.34       |        8.78       |
| St. Louis (MO)        |   8848  |  24267  |       2.74       |        7.29       |
| San Francisco (CA)    |   9577  |  26694  |       2.79       |        9.95       |
| Corona (CA)           |   3724  |   9257  |       2.49       |        6.87       |

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

def read_npz(path):
    with np.load(path, allow_pickle=True) as f:
        return parse_npz(f)


def parse_npz(f):
    x = torch.from_numpy(f['x']).to(torch.float)
    y = torch.from_numpy(f['y']).to(torch.long)
    edge_attr = torch.from_numpy(f['edge_attr']).to(torch.float)
    edge_index = torch.from_numpy(f['edge_index']).to(torch.long).t().contiguous()
    edge_attr_dir = torch.from_numpy(f['edge_attr_dir']).to(torch.float)
    edge_attr_ang = torch.from_numpy(f['edge_attr_ang']).to(torch.float)
    coords = torch.from_numpy(f['coordinates']).to(torch.float)
    severity_labels = torch.from_numpy(f['severity_labels']).to(torch.long)
    return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, 
                edge_attr_dir=edge_attr_dir, edge_attr_ang=edge_attr_ang, 
                coords=coords, severity_labels=severity_labels)


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

In addition, raw features can be found in our [Google Drive](https://drive.google.com/drive/folders/1dmWRkFhZvIjiMAeLNMzZsylI5i6i7bur?usp=sharing)
```python
# Raw node features include 'y', 'x', 'street_count', 'highway', 'ref', 'geometry', 'accident_cnt', 'severity', 'start_time', 'end_time'
with open('your_file_path/'+city_name+'_raw_node_attr.pkl', 'rb') as fp:
    gdf_nodes_raw = pickle.load(fp)

# Raw edge features include 'osmid', 'oneway', 'name', 'highway', 'length', 'ref', 'lanes', 'geometry', 'bridge', 'maxspeed', 'access', 'tunnel', 'junction'
with open('your_file_path/'+city_name+'_raw_edge_attr.pkl', 'rb') as fp:
    gdf_edges_raw = pickle.load(fp)
```
