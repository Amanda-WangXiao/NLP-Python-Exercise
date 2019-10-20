
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True, limit=300000)

dist1 = np.dot(model['man'], model['woman'])/(np.linalg.norm(model['man'])*np.linalg.norm(model['woman']))
print("余弦距离为：\t"+str(dist1))

