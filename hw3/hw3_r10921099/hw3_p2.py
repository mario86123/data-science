from genetic_selection import GeneticSelectionCV
from sklearn import linear_model
import numpy as np

with open('hw3_Data1/gene.txt', 'r') as f:
    content = f.read()
    gene = np.array(content.split()).astype(float).reshape((2000,62)).T
    
with open('hw3_Data1/index.txt', 'r') as f:
    index = f.read().splitlines()
    del index[-1]
    
with open('hw3_Data1/label.txt', 'r') as f:
    content = f.read()
    label = np.array(content.split()).astype(int)
    label = [int(i > 0) for i in label]

estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")
selector = GeneticSelectionCV(estimator,
                              cv=5,
                              verbose=0,
                              scoring="accuracy",
                              max_features=10,
                              n_population=50,
                              crossover_proba=0.5,
                              mutation_proba=0.2,
                              n_generations=40,
                              crossover_independent_proba=0.5,
                              mutation_independent_proba=0.05,
                              tournament_size=3,
                              n_gen_no_change=10,
                              caching=True,
                              n_jobs=-1)
selector = selector.fit(gene, label)

for i in range(len(selector.support_)):
    if selector.support_[i] == True:
        print([index[i].split()[0]])