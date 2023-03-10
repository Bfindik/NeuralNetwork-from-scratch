import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import train_multilayer
import matplotlib.pyplot as plt
df = pd.read_csv('Iris.csv')
column_name = "Species"
targets = np.array(df[column_name])
df.drop(columns=[column_name], inplace=True)
learning_rate = 0.01
input_vectors=np.array(df)

X_train, X_test, y_train, y_test = train_test_split(input_vectors, targets, test_size = .2,random_state=42)
nn = train_multilayer.neural_network(learning_rate)
nn.train(X_train,y_train,500)
def test(nn,input_vectors,targets):#buradaki input_vectors ve targets test için ayrıdığımız veriler
    # Loop through all the instances to measure the error
    error = 0
    outputs=[]#çıkan tahminler için dizi 
    for data_instance_index in range(0,len(input_vectors)):
        data_point = input_vectors[data_instance_index]
        target = targets[data_instance_index]
        prediction = nn.forward_prop(data_point)#tahmin yapma
        """tüm test örnekleri için predict işlemi yapmak ve çikan sonuçlara 
        yuvarlayarak sonuca varmak"""
        if prediction<0.5:
            print("Iris-setosa")
            outputs.append(0)
        elif prediction<1.45:
            print("Iris-versicolor")
            outputs.append(1)
        elif prediction<3:
            print("Iris-virginica")
            outputs.append(2)
        error += np.square(prediction - target)
    mean_squared_error = error / len(input_vectors)#modelin ortalama hatası(mean squared_error) hesaplanır
    """tahmin edilen değeler ile gerçek değerler arasindaki farki gözlemlemek
    içn mean squared erroru bulunuyor."""
    return mean_squared_error,outputs
mean_squared_error,outputs= test(nn,X_test,y_test)
def accuracy_test(y_test,outputs):#doğruluk ölçme 
    sayac1=sayac2=0#doğru ve yanlış prediction için sayaçlar
    for i in range(len(outputs)):
        if outputs[i]==y_test[i]:
            sayac1+=1
        else:
            sayac2+=1
    return sayac1,sayac2
sayac1,sayac2=accuracy_test(y_test,outputs)
print(sayac1,"true",sayac2,"false")

