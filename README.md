# TRAVEL

## Introduction

<!-- This is an implementation of Traffic Accident Vulnerability Estimation via Linkage (TRAVEL), a graph neural network framework proposed in the following paper: -->
This repo contains the graph-based traffic accident datasets proposed in the following paper:

Traffic Accident Prediction using Graph Neural Networks: New Datasets and the TRAVEL Model. Baixiang Huang, Bryan Hooi.

<!-- Please cite our paper if you use the datasets in this repo. -->

## Datasets

We build the datasets based on real-world road data from [OpenStreetMap](https://www.openstreetmap.org/) and the US-Accident datasets [(Moosavi et al., 2019)](https://arxiv.org/abs/1909.09638). Datasets are available in the directory `dataset`.

Currently, we have released datasets for 1,000 US cities. The table below shows the statistics of 100 major US cities.

These benchmark datasets can be used for two kind of tasks: traffic accident occurrence prediction and accident severity prediction. For the accident occurrence prediction task, we use binary labels to indicate whether a node contains at least one accident. For the severity prediction task, severity is represented by a number between 0 and 7, where 0 denotes no accident, 1 indicates the most negligible impact on traffic, and 7 indicates a significant impact on traffic. All environmental features obtained from OpenStreetMap are preprocessed. 

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
| Fort Worth (TX)       |  29846  |  77583  |        2.6       |        5.11       |
| Kansas City (MO)      |  22175  |  56955  |       2.57       |        6.02       |
| Philadelphia (PA)     |  24870  |  61518  |       2.47       |        5.32       |
| Ontario (CA)          |   3976  |  10127  |       2.55       |        8.73       |
| Birmingham (AL)       |  11298  |  31302  |       2.77       |       10.09       |
| Memphis (TN)          |  23986  |  63006  |       2.63       |        9.12       |
| Bakersfield (CA)      |  11959  |  29276  |       2.45       |        5.28       |
| Spartanburg (SC)      |   1853  |   4863  |       2.62       |       14.03       |
| Augusta (GA)          |   9395  |  23450  |        2.5       |       21.05       |
| Cincinnati (OH)       |   8190  |  20287  |       2.48       |       15.21       |
| New Orleans (LA)      |  15266  |  39961  |       2.62       |       15.99       |
| Whittier (CA)         |   1846  |   4860  |       2.63       |        8.56       |
| Lafayette (LA)        |   6054  |  15882  |       2.62       |       27.65       |
| Milwaukee (WI)        |  14382  |  38866  |        2.7       |       12.37       |
| Fresno (CA)           |  14968  |  38189  |       2.55       |        5.94       |
| Saint Petersburg (FL) |   9246  |  26709  |       2.89       |       17.72       |
| Hialeah (FL)          |   3542  |  10229  |       2.89       |        7.14       |
| Hayward (CA)          |   3345  |   8187  |       2.45       |        9.9        |
| Fort Myers (FL)       |   3160  |   7887  |        2.5       |       11.39       |
| Santa Clarita (CA)    |   6801  |  15073  |       2.22       |        4.88       |
| Pittsburgh (PA)       |   9176  |  23818  |        2.6       |       14.22       |
| Boston (MA)           |  11007  |  25350  |        2.3       |       20.28       |
| Shreveport (LA)       |   8591  |  23059  |       2.68       |       23.69       |
| York (PA)             |   1226  |   3497  |       2.85       |       44.37       |
| Lancaster (PA)        |   788   |   2043  |       2.59       |       62.18       |
| Baldwin Park (CA)     |   1096  |   2663  |       2.43       |       14.05       |
| Cleveland (OH)        |   9029  |  24873  |       2.75       |        17.6       |
| West Palm Beach (FL)  |   2987  |   7792  |       2.61       |       21.39       |
| Anderson (SC)         |   1682  |   4646  |       2.76       |       14.03       |
| Washington (DC)       |   9928  |  26850  |        2.7       |       18.32       |
| Tracy (CA)            |   2786  |   6714  |       2.41       |        4.38       |
| Irvine (CA)           |   7779  |  17651  |       2.27       |        5.54       |
| Sarasota (FL)         |   2529  |   6574  |        2.6       |       10.68       |
| Chattanooga (TN)      |   9327  |  24444  |       2.62       |       14.72       |
| Gardena (CA)          |   827   |   2253  |       2.72       |        12.7       |
| Bradenton (FL)        |   1790  |   4854  |       2.71       |        9.66       |
| Fremont (CA)          |   6934  |  16014  |       2.31       |        5.7        |
| Stockton (CA)         |   7437  |  19555  |       2.63       |        7.21       |
| Providence (RI)       |   4688  |  12904  |       2.75       |        8.06       |
| Colorado Springs (CO) |  20223  |  49956  |       2.47       |        7.61       |
| Pomona (CA)           |   2882  |   7480  |        2.6       |       12.42       |
| Hollywood (FL)        |   3758  |  10203  |       2.72       |        9.39       |
| Pensacola (FL)        |   3494  |   9950  |       2.85       |        8.21       |
| Baltimore (MD)        |  12543  |  31962  |       2.55       |       10.16       |
| Elgin (IL)            |   2915  |   8041  |       2.76       |       31.56       |
| Tempe (AZ)            |   4708  |  11890  |       2.53       |       14.93       |
| Norfolk (VA)          |   8454  |  22123  |       2.62       |        8.59       |
| Santa Rosa (CA)       |   5477  |  13673  |        2.5       |        6.48       |
| Livermore (CA)        |   3509  |   8686  |       2.48       |        5.67       |
| Myrtle Beach (SC)     |   2612  |   6692  |       2.56       |        8.27       |

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
- `edge_attr_dir`: Directional edge feature matrix with shape `[num_edges, 2]`
- `edge_attr_ang`: Angular edge feature matrix with shape `[num_edges, 3]`
- `coords`: Node coordinates
- `edge_index`: Graph indices in COO format (coordinate format) with shape `[2, num_edges]` and type torch.long

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
