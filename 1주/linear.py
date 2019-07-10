import keras
import numpy as np
import random
import matplotlib.pyplot as plt
testsize = 10
list1 = [561, 845, 154, 326, 356, 34, 6, 3, 64, 324, 5, 37, 73, 23, 543, 23, 58, 16, 99, 23, 143, 2, 33, 63, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 23, 523, 12,
         521, 56, 623, 23, 52, 12, 0, 23, 42, 64, 23, 42, 12, 5, 2, 42, 21, 53, 2, 233, 231, 101, 1131, 2314, 320]
x_train = np.array(list1, dtype=np.float32)
y_train = x_train+10

# network
model = keras.models.Sequential()
model.add(keras.layers.Dense(
    10, input_shape=(1,), activation='linear'))
model.add(keras.layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse')  # mean squred error
model.fit(x_train, y_train, batch_size=1, epochs=200)

# test
test_x = [0]
for a in range(testsize):
    a = random.randrange(1, 1000)
    test_x.append(a)

model.summary()
# rmse???
x_test = np.array(test_x, dtype=np.float32)
y_test = x_test+10
y_pre = model.predict(x_test).flatten()
S = np.float32(0)
for i in range(testsize):
    print('x :'+str(x_test[i]) + '\t y_pred : '+str(y_pre[i]))
    S += pow(y_test[i]-y_pre[i], 2)
mse = S/testsize
print('cost : ' + str(pow(mse, 0.5)))
print('----------------')

plt.plot(x_train, model.predict(x_train), 'b', x_train, y_train, 'k')
plt.scatter(x_test, y_pre)
plt.show()
