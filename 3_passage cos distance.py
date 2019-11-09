import math
from gensim.models.keyedvectors import KeyedVectors
import pkuseg

model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True, limit=300000)

seg = pkuseg.pkuseg()
sentence_1 = ''
sentence_2 = ''

text1 = seg.cut(sentence_1)  # 进行分词
text2 = seg.cut(sentence_2)


stopwords = [line.strip() for line in open('Englishstopwords.txt', encoding='UTF-8').readlines()]
clean1= list()
clean2= list()
for word in sentence_1.split(" "):
    if word not in stopwords:
        clean1.append(word)


for word in sentence_2.split(" "):
    if word not in stopwords:
        clean2.append(word)

vec1=list()
vec2=list()
sum=0
sq1=0
sq2=0
length=len(clean1+clean2)
for i in range(length):
        vec1.append(model[clean1[i]])
        vec2.append(model[clean2[i]])
        sum = sum+vec1[i] * vec2[i]
        sq1 = sq1+pow(vec1[i], 2)
        sq2 = sq2+pow(vec2[i], 2)
dist1 = result = round(float(sum) / (math.sqrt(sq1) * math.sqrt(sq2)), 2)
print("余弦距离为：\t"+str(dist1))