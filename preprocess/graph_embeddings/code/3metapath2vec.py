import os
import sys
import pickle
import datetime
import pandas as pd
from gensim.models import Word2Vec
from stellargraph import StellarGraph
from stellargraph.data import UniformRandomMetaPathWalk

#python3 metapath2vec.py ../results/graph/ ../results/graph/metapaths/ 32 30 3 ../results/metapath2vec/ number ../../hierarq/data/attrib_vecs/categ_priority/categ_priority-not_filtered.csv

GRAPHS_DIR = sys.argv[1]
METAPATHS_DIR = sys.argv[2]
E_DIMENSIONS = [int(d) for d in sys.argv[3].split(',')]
WALK_LENGTH = int(sys.argv[4])
WINDOW_SIZE = int(sys.argv[5])
RES_PATH = sys.argv[6]
INDEX_COL = sys.argv[7]
LOG = pd.read_csv(sys.argv[8], usecols=[INDEX_COL])

RANDOM_WALKS_PER_NODE = 10

graphs = [file for file in os.listdir(GRAPHS_DIR) if '.pickle' in file and 'het' in file]
print(f'{len(graphs)} graphs will be processed!')
print(f'Vector dimensions to be generated for each graph: {E_DIMENSIONS}')
print(f'Metapath2Vec parameters: walk_length={WALK_LENGTH}, window={WINDOW_SIZE}')
print(f'Results will be saved to {RES_PATH}')

for i, graph_file in enumerate(graphs):
	print(f'\n\n===========File {i+1}: {graph_file}===========')

	#Load graph
	with open(f'{GRAPHS_DIR}{graph_file}','rb') as f:
		graph = pickle.load(f)

	with open(f'{METAPATHS_DIR}{graph_file}','rb') as f:
		metapaths = pickle.load(f)

	for dimension in E_DIMENSIONS:
		print(f'Vector dimensions = {dimension}')
		print(f'\n{datetime.datetime.now()}: Generating walks...')
		rw = UniformRandomMetaPathWalk(graph, seed=1)
		walks = rw.run(
		    nodes=list(graph.nodes()),  # root nodes
		    length=WALK_LENGTH,  # maximum length of a random walk
		    n=RANDOM_WALKS_PER_NODE,  # number of random walks per root node
		    metapaths=metapaths,  # the metapaths
		)

		print(f'{datetime.datetime.now()}: Done! Calculating embbeding model...')
		model = Word2Vec(walks, vector_size=dimension, window=WINDOW_SIZE, seed=1)
				

		filename = f"metapath_{dimension}d_{WINDOW_SIZE}w_{WALK_LENGTH}l-{graph_file.split('/')[-1].replace('.pickle','')}"
		model.save(f'{RES_PATH}models/{filename}.model')

		model = Word2Vec.load(f'{RES_PATH}models/{filename}.model')
		process_cases = list(LOG[INDEX_COL].unique())
		vectors = [list(model.wv.get_vector(c)) for c in process_cases]
		vectors_df = pd.DataFrame(vectors)
		vectors_df[INDEX_COL] = pd.Series(process_cases)
		vectors_df.to_csv(f'{RES_PATH}{filename}.csv', index=False)
		print(f'{datetime.datetime.now()}: Done!')