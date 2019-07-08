from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical
import numpy
(train_img, train_labels),(test_img, test_labels)=mnist.load_data()
network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))
network.compile(optimizer = 'rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
train_img = train_img.reshape((60000, 28*28))
train_img = train_img.astype('float32')/255
test_img = test_img.reshape((10000,28*28))
test_img = test_img.astype('float32')/255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
network.fit(train_img, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_img, test_labels)
print('test_acc:',test_acc)