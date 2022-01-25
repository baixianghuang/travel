# TRAVEL

## Introduction

<!-- This is an implementation of Traffic Accident Vulnerability Estimation via Linkage (TRAVEL), a graph neural network framework proposed in the following paper: -->
This repo contains the graph-based traffic accident datasets proposed in the following paper:

[Traffic Accident Prediction using Graph Neural Networks: New Datasets and the TRAVEL Model].
Baixiang Huang, Bryan Hooi.

<!-- Please cite our paper if you use the datasets in this repo. -->

## Datasets

We build the datasets based on real-world road data from [OpenStreetMap](https://www.openstreetmap.org/) and the US-Accident datasets [(Moosavi et al., 2019)](https://arxiv.org/abs/1909.09638). Datasets are available in the directory `dataset_travel`.

Currently, we have released datasets for more than 500 US cities include Los Angeles (CA), Houston (TX), Charlotte (NC), Dallas (TX), Austin (TX), Miami (FL), Raleigh (NC), Nashville (TN), New York City (NY), Atlanta (GA), Baton Rouge (LA), Orlando (FL), Oklahoma City (OK), Sacramento (CA), Phoenix (AZ), Minneapolis (MN), San Diego (CA), Seattle (WA), San Antonio (TX), Saint Paul (MN), Jacksonville (FL), Richmond (VA), Portland (OR), San Jose (CA), Indianapolis (IN), Greenville (SC), Columbia (SC), Chicago (IL), Denver (CO), Tucson (AZ), Omaha (NE), Tulsa (OK), Tampa (FL), Rochester (NY), Dayton (OH), Detroit (MI), Flint (MI), Oakland (CA), Riverside (CA), Fort Lauderdale (FL), Grand Rapids (MI), Louisville (KY), Long Beach (CA), Columbus (OH), Salt Lake City (UT), El Paso (TX), Anaheim (CA), St. Louis (MO), San Francisco (CA), Corona (CA), Fort Worth (TX), Kansas City (MO), Philadelphia (PA), Ontario (CA), Birmingham (AL), Memphis (TN), Bakersfield (CA), Spartanburg (SC), Augusta (GA), Cincinnati (OH), New Orleans (LA), Whittier (CA), Lafayette (LA), Milwaukee (WI), Fresno (CA), Saint Petersburg (FL), Hialeah (FL), Hayward (CA), Fort Myers (FL), Santa Clarita (CA)...

These benchmark datasets can be used for two kind of tasks: traffic accident occurrence prediction and accident severity prediction. For the accident occurrence prediction task, we use binary labels to indicate whether a node contains at least one accident. For the severity prediction task, severity is represented by a number between 0 and 7, where 0 denotes no accident, 1 indicates the most negligible impact on traffic, and 7 indicates a significant impact on traffic. All environmental features obtained from OpenStreetMap are preprocessed. 

The table below shows the OpenStreetMap features included in our datasets.

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
- `edge_attr_dir`: Directional edge feature matrix with shape [num_edges, 2]
- `edge_attr_ang`: Angular edge feature matrix with shape [num_edges, 3]
- `coords`: Node coordinates
- `edge_index`: Graph indices in COO format (coordinate format) with shape [2, num_edges] and type torch.long

<!--  You can use the code below to load the datasets.
```python
class TRAVELDataset(InMemoryDataset):
    r"""The TRAVEL datasets from the
    `". Traffic Accident Prediction using Graph Neural Networks: New Datasets and the TRAVEL Model" 
    <https://link>`_ paper.
    Nodes represent intersections or dead-end nodes and edges represent roads.
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
``` -->

In addition, raw features can be found in our [Google Drive](https://drive.google.com/drive/folders/1dmWRkFhZvIjiMAeLNMzZsylI5i6i7bur?usp=sharing)
```python
# Raw node features include 'y', 'x', 'street_count', 'highway', 'ref', 'geometry', 'accident_cnt', 'severity', 'start_time', 'end_time'
with open('your_file_path/'+city_name+'_raw_node_attr.pkl', 'rb') as fp:
    gdf_nodes_raw = pickle.load(fp)

# Raw edge features include 'osmid', 'oneway', 'name', 'highway', 'length', 'ref', 'lanes', 'geometry', 'bridge', 'maxspeed', 'access', 'tunnel', 'junction'
with open('your_file_path/'+city_name+'_raw_edge_attr.pkl', 'rb') as fp:
    gdf_edges_raw = pickle.load(fp)
```
