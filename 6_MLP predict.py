from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pandas as pd

model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True, limit=300000)

clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(5, 2),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=200, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)

my_data = pd.read_csv('test_truth.txt',
                        delim_whitespace = True,header = None,
                        names = ['x','y'])

x = my_data.x
y = my_data.y

sentences=''
passtest=np.array(model[sentences].reshape(-1,1))

x_train,x_test,y_train,y_test = train_test_split(x.values.reshape(-1,1),y.values.reshape(-1,1),train_size=0.1)
model = clf.fit(x_train,y_train)
print("预测结果：")
print(clf.predict(passtest))
print("训练数据上的准确度：%f"% (model.score(x_train,y_train)))
print("测试数据上的准确度: %f" %(model.score(x_test,y_test)))