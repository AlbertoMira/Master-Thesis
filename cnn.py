# Imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from keras import layers, models
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd

# Lateral
# Open the train data and their labels
pkl_file = open('Lateral_data.pkl', 'rb')
train_data= pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('Lateral_labels.pkl', 'rb')
labels= pickle.load(pkl_file)
pkl_file.close()


# Split the data and shuffle it. 80% train and 20% test.
X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2)
# Conver the numpy arrays to tensor.
X_train=tf.convert_to_tensor(X_train)
X_test=tf.convert_to_tensor(X_test)
y_train=tf.convert_to_tensor(y_train)
y_test=tf.convert_to_tensor(y_test)


# About 10% of the training data will be kept as validation set
x_val = X_train[-204:]
y_val = y_train[-204:]
X_train = X_train[:-204]
y_train = y_train[:-204]



# Create the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2],X_train.shape[3])))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((1, 1)))
model.add(Dropout(0.2))
model.add(layers.Conv2D(64, (1, 1), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(6))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model in rounds of 100 epochs
history = model.fit(X_train, y_train, epochs=100, 
                    validation_data=(x_val, y_val))

# Test the model
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)


# Save the model
model.save('Lateral_model.h5')


# Confusion matrix
a=model.predict(X_test)
print(np.argmax(a[0]))
print(y_test[0])

l_predictions=[]
i=0
while i<a.shape[0]:
  l_predictions.append(np.argmax(a[i]))
  i+=1

tf.math.confusion_matrix(y_test,l_predictions)
cm = confusion_matrix(y_true=y_test, y_pred=l_predictions)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Define the classes of the confusion matrix and plot it
l=['Initial position','Pole plant','Push','Pole release','Leg push','Sixth Pattern']
classes=np.array(l)
plot_confusion_matrix(cm=cm, classes=classes, title='Confusion Matrix')

# Plot the accuracy of train and validation set through epochs (before train the model, the number of epochs is set to 1000 to see when the validation accuracy does not
# improve anymore with more epochs
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(hist['epoch'], hist['accuracy'],
           label='Train accuracy')
  plt.plot(hist['epoch'], hist['val_accuracy'],
           label = 'Val accuracy')
  plt.ylim([0,1])
  plt.legend()
plot_history(history)





# Frontal
# Repeat the same process as with lateral, the only difference is the dataset (frontal train data and labels store in a pickle file)

pkl_file = open('Frontal_data.pkl', 'rb')
train_data= pickle.load(pkl_file)
pkl_file.close()


pkl_file = open('Frontal_labels.pkl', 'rb')
labels= pickle.load(pkl_file)
pkl_file.close()

X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2)
X_train=tf.convert_to_tensor(X_train)
X_test=tf.convert_to_tensor(X_test)
y_train=tf.convert_to_tensor(y_train)
y_test=tf.convert_to_tensor(y_test)

x_val = X_train[-201:]
y_val = y_train[-201:]
X_train = X_train[:-201]
y_train = y_train[:-201]

X_train.shape

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2],X_train.shape[3])))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((1, 1)))
model.add(Dropout(0.2))
model.add(layers.Conv2D(64, (1, 1), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(6))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, 
                    validation_data=(x_val, y_val))

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

a=model.predict(X_test)

l_predictions=[]
i=0
while i<a.shape[0]:
  l_predictions.append(np.argmax(a[i]))
  i+=1

tf.math.confusion_matrix(y_test,l_predictions)

cm = confusion_matrix(y_true=y_test, y_pred=l_predictions)

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



l=['Initial position','Pole plant','Push','Pole release','Leg push','Sixth Pattern']
classes=np.array(l)

plot_confusion_matrix(cm=cm, classes=classes, title='Confusion Matrix')

model.save('Frontal_model.h5')

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(hist['epoch'], hist['accuracy'],
           label='Train accuracy')
  plt.plot(hist['epoch'], hist['val_accuracy'],
           label = 'Val accuracy')
  plt.ylim([0,1])
  plt.legend()
plot_history(history)



