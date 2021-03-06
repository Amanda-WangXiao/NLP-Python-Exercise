from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets


clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(5, 2),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=200, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)

wine = datasets.load_wine()
x = wine.data
y = wine.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.1)
model = clf.fit(x_train,y_train)
print("训练数据上的准确度：%f"% (model.score(x_train,y_train)))
print("测试数据上的准确度: %f" %(model.score(x_test,y_test)))
