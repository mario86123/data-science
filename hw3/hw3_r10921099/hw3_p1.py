import numpy as np
from sklearn.feature_selection import SelectKBest, chi2

with open('hw3_Data1/gene.txt', 'r') as f:
    content = f.read()
    gene = np.array(content.split()).astype(float).reshape((2000,62)).T
    
with open('hw3_Data1/index.txt', 'r') as f:
    index = f.read().splitlines()
    del index[-1]
    
with open('hw3_Data1/label.txt', 'r') as f:
    content = f.read()
    label = np.array(content.split()).astype(int)
    label = [1 if i > 0 else -1 for i in label]
    
model = SelectKBest(chi2).fit(gene, label)
for i in model.get_support(indices=True):
    print([index[i].split()[0]])