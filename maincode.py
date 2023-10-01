"Import Libaries "
# import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pickle
#--------------------------------------------------------------------



print("KddCup Dataset")
print(" Process -  Attack Detection")


df= pd.read_csv('Kddcup.csv')
# df=df.iloc[:5000]
df.head(5)

df.describe()
#Check null values
df.isnull().sum()
#---------------------------------------------------------------

#importing the dataset
df_train_X= df
df_train_y=df_train_X["label"]
df_train_X=df_train_X.iloc[:,:20]
from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()

df_train_X['proto'] = number.fit_transform(df_train_X['proto'].astype(str))
df_train_X['service'] = number.fit_transform(df_train_X['service'].astype(str))
df_train_X['state'] = number.fit_transform(df_train_X['state'].astype(str))
#df_train_X['attack_cat'] = number.fit_transform(df_train_X['attack_cat'].astype(str))
print("==================================================")
print("KddCup Dataset")
print(" Preprocessing")
print("==================================================")

df_train_X.head(5)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(df_train_X,df_train_y,test_size = 0.30,random_state = 42)
print("X_train Shapes ",x_train.shape)
print("y_train Shapes ",y_train.shape)
print("x_test Shapes ",x_test.shape)
print("y_test Shapes ",y_test.shape)


from sklearn.ensemble import RandomForestClassifier

model= RandomForestClassifier(n_estimators = 10)  
model.fit(x_train, y_train)
rf_prediction = model.predict(x_test)
Result_3=accuracy_score(y_test, rf_prediction)*100
from sklearn.metrics import confusion_matrix

print()
print("---------------------------------------------------------------------")
print("Random Forest")
print()
print(metrics.classification_report(y_test,rf_prediction))
print()
print("Random Forest Accuracy is:",Result_3,'%')
print()
print("Confusion Matrix:")
cm2=confusion_matrix(y_test, rf_prediction)
print(cm2)
print("-------------------------------------------------------")
print()
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm2, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, rf_prediction)
plt.plot(fpr, tpr, marker='.', label='RF')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


#----------------------------------------------------------------
"lstm -2D expandedv"
x_train=np.expand_dims(x_train, axis=2)
x_test=np.expand_dims(x_test, axis=2)
y_train=np.expand_dims(y_train,axis=1)
y_test=np.expand_dims(y_test,axis=1)


"LSTM Algorithm "
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense,Conv1D,MaxPooling1D

nb_out = 1
model = Sequential()
model.add(LSTM(input_shape=(20, 1), units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=nb_out))
model.add(Activation("linear"))
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

print(model.summary())
# fit the model
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=1)
Result_3=model.evaluate(x_train,y_train,verbose=1)[1]*100
#from sklearn.metrics import accuracy_score
from sklearn import metrics

LSTM_prediction = model.predict(x_test)
from sklearn.metrics import confusion_matrix

print()
print("---------------------------------------------------------------------")
print(" LSTM")
print()
print(metrics.classification_report(y_test,LSTM_prediction.round()))
print()
print("LSTM  Accuracy is:",Result_3,'%')
print()
print("Confusion Matrix:")
cm2=confusion_matrix(y_test, LSTM_prediction.round())
print(cm2)
print("-------------------------------------------------------")
print()
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm2, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, LSTM_prediction.round())
plt.plot(fpr, tpr, marker='.', label='LSTM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_pred=rf_prediction
from easygui import *
Key = "Enter the intrusion  Id to be Search"
  
# window title
title = "  Fault Id "
# creating a integer box
str_to_search1 = enterbox(Key, title)
input = int(str_to_search1)

import tkinter as tk
if (y_pred[input] ==0 ):
    print("Non Attack ")
    root = tk.Tk()
    T = tk.Text(root, height=20, width=30)
    T.pack()
    T.insert(tk.END, "Non Attack ")
    tk.mainloop()
elif (y_pred[input] ==1 ):
    print("Attack ")
    root = tk.Tk()
    T = tk.Text(root, height=20, width=30)
    T.pack()
    T.insert(tk.END, "Attack ")
    tk.mainloop()
    import smtplib as smtp
    
    connection = smtp.SMTP_SSL('smtp.gmail.com', 465)
        
    email_addr = 'kkarthigadevikd@gmail.com'
    email_passwd = 'sbpq qgss wnda ptrs'
    connection.login(email_addr, email_passwd)
    connection.sendmail(from_addr=email_addr, to_addrs='karthigadevi241@gmail.com', msg="Attack kindly prevent IDS Atack ")
    connection.close()

filename = 'ids.pkl'
pickle.dump(model, open(filename, 'wb'))


