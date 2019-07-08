import keras
import numpy as np
x_test = np.array([101,1131,2314],dtype=np.float32)
y_test = np.array([101,1131,2314],dtype=np.float32)
lists = [99,23,143,0.1,33,63,1,2,3,4,5,6,7,8,9,10,0,23,523,12,521,56,623,64,23,42,12,5,2,42,21,53,2,233,231]
y_train=np.array(lists,dtype=np.float32)
x_train=np.array(lists,dtype=np.float32)
network = keras.models.Sequential()
network.add(keras.layers.Dense(len(lists)+1,input_dim=1,activation='relu'))
network.add(keras.layers.Dense(1,activation='relu'))
network.compile(optimizer = 'adam', loss='mean_squared_error')
network.fit(x_train, y_train,epochs=200, batch_size=11)
y_pre = network.predict(x_test).flatten()
S=np.float32(0)
for i in range(3):
    print('y : '+str(y_test[i])+', y_pred : '+str(y_pre[i]))
    S+=pow(y_test[i]-y_pre[i],2)
print('acc : '+ str(pow(S,0.5)/3))