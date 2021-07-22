import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from os import listdir

from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.optimizers import Adam

from keras.models import load_model

from keras.callbacks import ModelCheckpoint

df1 = pd.read_csv(<path to the train data csv file>)
df2 = pd.read_csv(<path to the train data csv file>)

df1.head()
df2.head()

df1.shape, df2.shape

path = <path to the csv file>
sequences = list()
for i in range(1,315):
    file_path = path + str(i) + '.csv'
    print(file_path)
    df = pd.read_csv(file_path, header=0)
    values = df.values
    sequences.append(values)

targets = pd.read_csv(<path to the test data csv file>)
targets = targets.values[:,1]

sequences[0]

groups = pd.read_csv(<path to known classes file>, header=0)
groups = groups.values[:,1]

len_sequences = []
for one_seq in sequences:
    len_sequences.append(len(one_seq))
pd.Series(len_sequences).describe

#Padding the sequence with the values in last row to max length
to_pad = 129
new_seq = []
for one_seq in sequences:
    len_one_seq = len(one_seq)
    last_val = one_seq[-1]
    n = to_pad - len_one_seq
   
    to_concat = np.repeat(one_seq[-1], n).reshape(4, n).transpose()
    new_one_seq = np.concatenate([one_seq, to_concat])
    new_seq.append(new_one_seq)
final_seq = np.stack(new_seq)

#truncate the sequence to length 60
from keras.preprocessing import sequence
seq_len = 60
final_seq=sequence.pad_sequences(final_seq, maxlen=seq_len, padding='post', dtype='float', truncating='post')

train = [final_seq[i] for i in range(len(groups)) if (groups[i]==2)]
validation = [final_seq[i] for i in range(len(groups)) if groups[i]==1]
test = [final_seq[i] for i in range(len(groups)) if groups[i]==3]

train_target = [targets[i] for i in range(len(groups)) if (groups[i]==2)]
validation_target = [targets[i] for i in range(len(groups)) if groups[i]==1]
test_target = [targets[i] for i in range(len(groups)) if groups[i]==3]
train = np.array(train)
validation = np.array(validation)
test = np.array(test)

train_target = np.array(train_target)
train_target = (train_target+1)/2

validation_target = np.array(validation_target)
validation_target = (validation_target+1)/2

test_target = np.array(test_target)
test_target = (test_target+1)/2

#Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(train,train_target)
predictins=logreg.predict(test)
print("Accuracy:",metrics.accuracy_score(test_target, predictions))
cnf_matrix = metrics.confusion_matrix(test_target, predictions)
cnf_matrix

#Random forest model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train, train_target);
predictions = rf.predict(test)
print("Accuracy:",metrics.accuracy_score(test_target, predictions))
cnf_matrix = metrics.confusion_matrix(test_target, predictions)
cnf_matrix

#Support vector machine model
from sklearn import svm
clf = svm.SVC(kernel='linear') # Linear Kernel
clf.fit(train, train_target)
predictions = clf.predict(test)
print("Accuracy:",metrics.accuracy_score(test_target, predictions))
cnf_matrix = metrics.confusion_matrix(test_target, predictions)
cnf_matrix

#RNN model
model = Sequential()
model.add(LSTM(256, input_shape=(seq_len, 4)))
model.add(Dense(1, activation='sigmoid'))

model.summary()

adam = Adam(lr=0.001)
chk = ModelCheckpoint('best_model.pkl', monitor='val_acc', save_best_only=True, mode='max', verbose=1)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit(train, train_target, epochs=200, batch_size=128, callbacks=[chk], validation_data=(validation,validation_target))

#loading the model and checking accuracy on the test data
model = load_model(<path to best model>)

from sklearn.metrics import accuracy_score
test_preds = model.predict_classes(test)
accuracy_score(test_target, test_preds)
