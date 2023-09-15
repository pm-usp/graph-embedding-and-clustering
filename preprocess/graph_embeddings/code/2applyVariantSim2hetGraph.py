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

SEQUENCES_PATH = '../../incident_evt_log-processed1-sequences.csv'
VARIANT_DISTANCE_MAX = 18
K = 10


def get_knn_seq(seq_line):
    def add_edge(x):
        if x[0] < VARIANT_DISTANCE_MAX:
            pair_key = f"variant#{seq_line['trace']}--variant#{unique_seq.iloc[x['index']]['trace']}"
            GRAPH_DATA['node_types']['knowledge_connector'].add(pair_key)
            GRAPH_DATA['sources'].append(pair_key)
            GRAPH_DATA['targets'].append(f"variant#{seq_line['trace']}")
            GRAPH_DATA['sources'].append(pair_key)
            GRAPH_DATA['targets'].append(f"variant#{unique_seq.iloc[x['index']]['trace']}") 

    #dists = unique_seq.apply(lambda x: get_d(seq_line['trace'],x['trace']), axis=1)
    dists = unique_seq.apply(lambda x: editdistance.eval(seq_line['trace'], x['trace']), axis=1).reset_index().sort_values(by=0)
    dists.iloc[1:K+1].apply(lambda x: add_edge(x), axis=1)


graphs = [filename for filename in os.listdir(GRAPHS_DIR) if 'het' in filename and 'variant' in filename and 'variants_edges' not in filename and '.pickle' in filename]
unique_seq = pd.read_csv(SEQUENCES_PATH).groupby('trace').count().reset_index().rename(columns={'number':'trace_qtt'})
print(f'{len(graphs)} graphs to be processed')
for i, graph_filename in enumerate(graphs):
    print(f'\n\n====== File {i+1}: {graph_filename} ======')
    print('Original graph info:')
    graph = pickle.load(open(f'{GRAPHS_DIR}{graph_filename}','rb'))
    print(graph.info())

    print('\nAdding edges among similar variants ...')
    original_graph_data = pickle.load(open(f'{GRAPHS_DATA_DIR}{graph_filename}','rb'))
    GRAPH_DATA = {'sources': list(original_graph_data['edges'].source),
                  'targets': list(original_graph_data['edges'].target),
                  'weights': original_graph_data['weights'],
                  'node_types': original_graph_data['node_types']}
    GRAPH_DATA['node_types']['knowledge_connector'] = set()              
    _ = unique_seq.apply(get_knn_seq, axis=1)
    print('Done!')

    filename = f'{graph_filename.replace(".pickle","")}_{K}nn_variants_edges_{VARIANT_DISTANCE_MAX}max.pickle'

    edges = pd.DataFrame({'source': GRAPH_DATA['sources'], 'target': GRAPH_DATA['targets']})
    GRAPH_DATA['node_types']['knowledge_connector'] = pd.DataFrame(index=list(GRAPH_DATA['node_types']['knowledge_connector']))
    pickle.dump({'node_types': GRAPH_DATA['node_types'], 'edges': edges}, open(f'{RESULTS_GRAPH_DATA_PATH}{filename}','wb'))

    graph = StellarGraph(GRAPH_DATA['node_types'], edges)
    pickle.dump(graph, open(f'{RESULTS_GRAPH_PATH}{filename}','wb'))
    print('\nNew graph info:')
    print(graph.info())

    metapaths = pickle.load(open(f'{METAPATHS_DATA_DIR}{graph_filename}','rb'))
    metapaths.append(['case', 'variant', 'knowledge_connector', 'variant','case'])
    pickle.dump(metapaths, open(f'{RESULTS_METAPATH_PATH}{filename}','wb'))