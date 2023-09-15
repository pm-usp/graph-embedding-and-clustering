import pickle 
import pandas as pd
from stellargraph import StellarGraph

RESULTS_PATH = '../results/graph/'
EVENT_LOG_PATH = '../../incident_evt_log-processed1-classiclogformat.csv'
SEQUENCES_PATH = '../../incident_evt_log-processed1-sequences.csv'

VARIANT_WEIGHT = 1
ATTRIB_WEIGHT = 1
ACTIVITY_WEIGHT = 1

log = pd.read_csv(EVENT_LOG_PATH, index_col = 0)
seq = pd.read_csv(SEQUENCES_PATH)
df = log.merge(seq).fillna('')

sources = []
targets = []
weights = []
cases = set()
activities = set()
attribs = set()
variants = set()

def create_edges(event):
  cases.add(event['number'])

  sources.append(event['number'])
  targets.append('activity#'+event['activity'])
  weights.append(ACTIVITY_WEIGHT)
  activities.add('activity#'+event['activity'])

  sources.append(event['number'])
  targets.append('category#'+event['category'])
  weights.append(ATTRIB_WEIGHT)
  attribs.add('category#'+event['category'])

  sources.append(event['number'])
  targets.append('priority#'+event['priority'])
  weights.append(ATTRIB_WEIGHT)
  attribs.add('priority#'+event['priority'])

  sources.append(event['number'])
  targets.append('variant#'+event['trace'])
  weights.append(VARIANT_WEIGHT)
  variants.add('variant#'+event['trace'])

_ = df.apply(create_edges, axis = 1)
edges = pd.DataFrame({'source': sources, 'target': targets})

# ############## ACT, CATEG, PRIOR ############## 
# print(f'Creating act and attrib graph...')
# node_types = {"case": pd.DataFrame(index=list(cases)),
#               "activity": pd.DataFrame(index=list(activities)),
#               "attrib": pd.DataFrame(index=list(attribs)), 
#               }
# filtered_edges = edges[~edges.target.str.contains('variant#')].drop_duplicates()
# graph = StellarGraph(node_types, filtered_edges)

# filename = 'hetgraph_act_attrib.pickle'
# pickle.dump(graph, open(f'{RESULTS_PATH}{filename}','wb'))
# pickle.dump({'node_types': node_types, 'edges': filtered_edges}, open(f'{RESULTS_PATH}/hetgraph_data/{filename}','wb'))

# metapaths = [['case','activity','case'], ['case','attrib','case']]
# pickle.dump(metapaths, open(f'{RESULTS_PATH}metapaths/{filename}','wb'))
# ###############################################


# ################ ACT, CATEG ################## 
# print(f'Creating act and categ graph...')
# node_types = {"case": pd.DataFrame(index=list(cases)),
#               "activities": pd.DataFrame(index=list(activities)),
#               "attrib": pd.DataFrame(index=list(attribs)), 
#               }
# filtered_edges = edges[(~edges.target.str.contains('variant#')) & (~edges.target.str.contains('priority#'))].drop_duplicates()
# graph = StellarGraph(node_types, filtered_edges)

# filename = 'hetgraph_act_categ.pickle'
# pickle.dump(graph, open(f'{RESULTS_PATH}{filename}','wb'))
# pickle.dump({'node_types': node_types, 'edges': filtered_edges}, open(f'{RESULTS_PATH}/hetgraph_data/{filename}','wb'))

# metapaths = [['case','activity','case'], ['case','attrib','case']]
# pickle.dump(metapaths, open(f'{RESULTS_PATH}metapaths/{filename}','wb'))
# ###############################################


############# VARIANT, ATTRIBS ############### 
# print(f'Creating variant and categ graph...')
# node_types = {"case": pd.DataFrame(index=list(cases)),
#               "variant": pd.DataFrame(index=list(variants)),
#               "attrib": pd.DataFrame(index=list(attribs)), 
#               }
# filtered_edges = edges[(~edges.target.str.contains('activity#'))].drop_duplicates()
# graph = StellarGraph(node_types, filtered_edges)

# filename = f'hetgraph_variant{VARIANT_WEIGHT}_attrib{ATTRIB_WEIGHT}.pickle'
# pickle.dump(graph, open(f'{RESULTS_PATH}{filename}','wb'))
# pickle.dump({'node_types': node_types, 'edges': filtered_edges}, open(f'{RESULTS_PATH}/hetgraph_data/{filename}','wb'))

# metapaths = [['case','variant','case'], ['case','attrib','case']]
# pickle.dump(metapaths, open(f'{RESULTS_PATH}metapaths/{filename}','wb'))
###############################################


############### VARIANT, CATEG ############### 
print(f'Creating variant and categ graph...')
node_types = {"case": pd.DataFrame(index=list(cases)),
              "variant": pd.DataFrame(index=list(variants)),
              "attrib": pd.DataFrame(index=list(attribs)), 
              }
filtered_edges = edges[(~edges.target.str.contains('activity#')) & (~edges.target.str.contains('priority#'))].drop_duplicates()
graph = StellarGraph(node_types, filtered_edges)

filename = f'hetgraph_variant{VARIANT_WEIGHT}_categ{ATTRIB_WEIGHT}.pickle'
pickle.dump(graph, open(f'{RESULTS_PATH}{filename}','wb'))
pickle.dump({'node_types': node_types, 'edges': filtered_edges}, open(f'{RESULTS_PATH}/hetgraph_data/{filename}','wb'))

metapaths = [['case','variant','case'], ['case','attrib','case']]
pickle.dump(metapaths, open(f'{RESULTS_PATH}metapaths/{filename}','wb'))
###############################################


                      



