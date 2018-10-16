from keras import layers, models, optimizers, metrics, losses
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


print("train_images.shape", train_images.shape)
print("image shape", train_images[0].shape)
print("train_labels.shape", train_labels.shape)
print("len(train_labels)", len(train_labels))

print("test_images.shape", test_images.shape)
print("image shape", test_images[0].shape)
print("test_labels.shape", test_labels.shape)
print("len(test_labels)", len(test_labels))

train_labels[0:10]

#import matplotlib.pyplot as plt
#plt.imshow(train_images[0], cmap='gray')
#plt.show()


ori_train_images = train_images.copy()
ori_train_labels = train_labels.copy()

ori_test_images = test_images.copy()
ori_test_labels = test_labels.copy()


net = models.Sequential()
net.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
net.add(layers.Dense(10, activation='softmax')) # 10 outputs for [0, 9]

net.compile(optimizer=optimizers.RMSprop(), 
            loss=losses.categorical_crossentropy,
            metrics=[metrics.categorical_accuracy])


train_images = train_images.reshape((60000, 28 * 28)) # Reshape an image to [1, 28 * 28]
train_images = train_images.astype("float32") / 255 # Scale to [0, 1]

test_images = test_images.reshape((10000, 28 * 28)) # Reshape an image to [1, 28 * 28]
test_images = test_images.astype("float32") / 255 # Scale to [0, 1]

from keras.utils import to_categorical

train_labels = to_categorical(ori_train_labels)
test_labels = to_categorical(ori_test_labels)


print("train_images.shape", train_images.shape)
print("image shape", train_images[0].shape)
print("train_labels.shape", train_labels.shape)
print("train_labels\n", train_labels)
print("train_labels[0]", train_labels[0])

history = net.fit(train_images, 
                  train_labels, 
                  epochs=10,
                  batch_size=128,
                  validation_data=(test_images, test_labels))

results = net.evaluate(test_images, test_labels)
print(results)


import numpy as np
results = net.predict(test_images)

print("1st predict results: \n", results[0])
print("1st predict label: \n", np.argmax(results[0]))