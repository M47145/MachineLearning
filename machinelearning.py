import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix
import pickle # save encoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import seaborn as sns



df = pd.read_csv('Phishing_Legitimate_full.csv')
df = df.drop(columns=['id'])

X=df.iloc[:,:48]
y=df.loc[:,['CLASS_LABEL']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)



model = Sequential()
model.add(Dropout(rate=0.1))
model.add(Dense(units=12, input_dim=X.shape[1],activation='relu'))
model.add(Dense(units=6, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
history=model.fit(X_train, y_train, epochs=100, batch_size=32,
                   validation_data=(X_test, y_test))



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


y_pred = model.predict(X_test)


y_pred_prob = y_pred
y_pred_prob = y_pred_prob.flatten()
y_pred = np.where(y_pred > 0.5, 1, 0)


# metriikat
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
ps = precision_score(y_test, y_pred)
rs = recall_score(y_test, y_pred)
print(f'{cm}')
print(f'ac: {ac*100:.03f}%')
print(f'ps: {ps:.03f}')
print(f'rs: {rs:.03f}')
sns.heatmap(cm, annot=True, fmt='g')
plt.show()



model.save('phishing-model-ANN.h5')

# tallennetaan X skaaleri
with open('phishing-ANN-scaler_X.pickle', 'wb') as f:
    pickle.dump(scaler_X, f)







