import os
import pickle 
import editdistance
import pandas as pd
from stellargraph import StellarGraph

GRAPHS_DIR = '../results/graph/'
GRAPHS_DATA_DIR = '../results/graph/hetgraph_data/'
METAPATHS_DATA_DIR = '../results/graph/metapaths/'

RESULTS_GRAPH_PATH = '../results/graph/'
RESULTS_GRAPH_DATA_PATH = f'{RESULTS_GRAPH_PATH}/hetgraph_data/'
RESULTS_METAPATH_PATH = f'{RESULTS_GRAPH_PATH}/metapaths/'

EXPERT_NODE_NAME = 'expert_easy_categs_ml'

ML_CATEGS = ['Passwords and access', 'Remote access', 'Office pack']
def apply_categs_constr():
    GRAPH_DATA['node_types']['knowledge_connector'].add(EXPERT_NODE_NAME)
    for categ in ML_CATEGS:
        GRAPH_DATA['sources'].append(EXPERT_NODE_NAME)
        GRAPH_DATA['targets'].append(f"category#{categ}")


graphs = [filename for filename in os.listdir(GRAPHS_DIR) if 'het' in filename and 'expnode2categs' not in filename and '.pickle' in filename]
print(f'{len(graphs)} graphs to be processed')
for i, graph_filename in enumerate(graphs):
    print(f'\n\n====== File {i+1}: {graph_filename} ======')
    print('Original graph info:')
    graph = pickle.load(open(f'{GRAPHS_DIR}{graph_filename}','rb'))
    print(graph.info())

    GRAPH_DATA = pickle.load(open(f'{GRAPHS_DATA_DIR}{graph_filename}','rb'))
    GRAPH_DATA = {'sources': list(GRAPH_DATA['edges'].source),
                  'targets': list(GRAPH_DATA['edges'].target),
                  'node_types': GRAPH_DATA['node_types']}

    if 'knowledge_connector' not in GRAPH_DATA['node_types']:
        GRAPH_DATA['node_types']['knowledge_connector'] = set() 
    else: 
        GRAPH_DATA['node_types']['knowledge_connector'] = set(list(GRAPH_DATA['node_types']['knowledge_connector'].index))

    print('\nApplying categs must link contraints ...')          
    apply_categs_constr()
    print('Done!')

    filename = f'{graph_filename.replace(".pickle","")}_expnode2categs.pickle'

    edges = pd.DataFrame({'source': GRAPH_DATA['sources'], 'target': GRAPH_DATA['targets']})
    GRAPH_DATA['node_types']['knowledge_connector'] = pd.DataFrame(index=list(GRAPH_DATA['node_types']['knowledge_connector']))
    pickle.dump({'node_types': GRAPH_DATA['node_types'], 'edges': edges}, open(f'{RESULTS_GRAPH_DATA_PATH}{filename}','wb'))

    if 'variants' in GRAPH_DATA['node_types']:
        print(GRAPH_DATA['node_types']['variants'])
    graph = StellarGraph(GRAPH_DATA['node_types'], edges)
    pickle.dump(graph, open(f'{RESULTS_GRAPH_PATH}{filename}','wb'))
    print('\nNew graph info:')
    print(graph.info())

    metapaths = pickle.load(open(f'{METAPATHS_DATA_DIR}{graph_filename}','rb'))
    metapaths.append(['case', 'attrib', 'knowledge_connector', 'attrib','case'])
    pickle.dump(metapaths, open(f'{RESULTS_METAPATH_PATH}{filename}','wb'))