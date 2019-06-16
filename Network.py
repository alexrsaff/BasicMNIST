# Import necessary packages
import keras
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
import numpy

# Load and prepare training images and labels
train_images, train_labels = loadlocal_mnist(images_path="train-images-idx3-ubyte",labels_path="train-labels-idx1-ubyte")
new_labels = numpy.zeros((len(train_labels),10))
for pos,label in enumerate(train_labels):
    new_labels[pos][label]=1
train_labels = new_labels
train_images = train_images/255.0

# Build and train model
model = keras.Sequential()
model.add(keras.layers.Dense(units = 512, activation = "relu", input_shape = train_images[0].shape))
model.add(keras.layers.Dense(units = 10, activation = "softmax"))
model.compile(loss="categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])
model.fit(train_images,train_labels,epochs = 5)

# Load and prepare test images and labels
test_images, test_labels = loadlocal_mnist(images_path="t10k-images-idx3-ubyte",labels_path="t10k-labels-idx1-ubyte")
test_images = test_images/255.0

# Test model
predictions = numpy.argmax(model.predict(test_images), axis = 1)
for pos, prediction in enumerate(predictions):
    print("Prediction: ", prediction)
    print("Actual: ", test_labels[pos])
    plt.imshow(numpy.reshape(test_images[pos],(28,28)))
    plt.show()
    key = input("Enter x to exit or just enter to continue: ")
    if(key=='x'):
        break