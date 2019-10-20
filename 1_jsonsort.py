import json
count=0
m=open('test.txt','w')
l=[]
with open('News_Category_Dataset_v2.json', 'r') as f:
    for line in f:
        temp = json.loads(line)
        l.append(temp['category'])


a=sorted(l)
m.write(str(a))


