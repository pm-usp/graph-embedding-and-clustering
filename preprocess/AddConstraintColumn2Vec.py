import pandas as pd

#VEC_PATH = 'results/attrib_vecs/log_attrib_exp/'
#VEC_FILENAME = 'log_attrib_exp-not_filtered'
VEC_PATH = 'results/attrib_vecs/det_act_act_tfidf_attrib_vec/'
VEC_FILENAME = 'act_tfidf_attrib_vec_bin-det_act-not_filtered'

CONSTRAINT_FILE = 'constraints/easy_categs.csv'
RESULT_FILE = f'{VEC_PATH}{VEC_FILENAME}-with_constraint_col.csv'

print(f'\n\n#######################\nVectors file: {VEC_PATH}{VEC_FILENAME}.csv')
print(f'Constraint file: {CONSTRAINT_FILE}')
print(f'Result file: {RESULT_FILE}\n#######################')

constraints = pd.read_csv(CONSTRAINT_FILE)
print('\n\nConstraint:\n',constraints.head())
df = pd.read_csv(f'{VEC_PATH}{VEC_FILENAME}.csv', index_col = 0).merge(constraints)
print('\n\nVectors with constraint column:\n',df.head())

df.to_csv(RESULT_FILE, index = False)
