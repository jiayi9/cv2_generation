
PATH = "C:/LV_CHAO_IMAGE/simulation_data/"

input_shape = (256, 256,1)

################################################################################################################################

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import glob
import PIL
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os

os.chdir(PATH)

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        #print('root_dir:', root)  # 当前目录路径
        #print('sub_dirs:', dirs)  # 当前路径下所有子目录
        #print('files:', files)  # 当前路径下所有非目录子文件
        return dirs

pdts = file_name('.')

X = np.array([[]])
Y = np.array([[]])

for index, pdt in enumerate(pdts):
    print(index)
    print(pdt)
    files = glob.glob('./'+pdt+'/*.jpg')
    n = len(files)
    print(n)

    if pdt == "circle_256":
        X = np.array([
                np.array((Image.open(filename))) for filename in files
                ])
        Y = np.array([index for i in range(1,(n+1))])
        print(X.shape)
        print(Y.shape)
        
    else:        
        TMP_X = np.array([
            np.array((Image.open(filename))) for filename in files
        ])
        TMP_Y = np.array([index for i in range(1,(n+1))])
        print(TMP_X.shape)
        print(TMP_Y.shape)
        X = np.concatenate((X, TMP_X), axis = 0)
        Y = np.concatenate((Y, TMP_Y), axis = 0)

#import collections
#collections.Counter(Y)

#if K.image_data_format() == 'channels_last':
#    im_Ok = im_Ok.reshape(im_Ok.shape[0],img_rows, img_cols, 1)
#    im_NOK = im_NOK.reshape(im_NOK.shape[0],img_rows, img_cols, 1)
#    input_shape = (img_rows, img_cols, 1)
#else:
#    im_Ok = im_Ok.reshape(im_Ok.shape[0], 1,img_rows, img_cols)
#    im_NOK = im_NOK.reshape(im_NOK.shape[0], 1,img_rows, img_cols)
#    input_shape = (1,img_rows, img_cols)

X = np.expand_dims(X,3)
Y = np.expand_dims(Y,3)
n = len(X)  
    
# Sepreate train and test data
    
msk = np.random.rand(n) < 0.5

x_train = X[msk]
x_test = X[~msk]
y_train = Y[msk]
y_test = Y[~msk]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
num_classes = np.unique(Y).max() + 1
batch_size = 30
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

epochs = 30



# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

model.count_params()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


train_losses = model.history.history["loss"]
test_losses = model.history.history["val_loss"]

train_acc = model.history.history["acc"]
test_acc = model.history.history["val_acc"]

import matplotlib.pyplot as plt 
plt.plot(train_losses)
plt.plot(test_losses)
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


import matplotlib.pyplot as plt 
plt.plot(train_acc)
plt.plot(test_acc)
plt.title('Model Overall Accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



from sklearn.metrics import classification_report, confusion_matrix

print('Confusion Matrix')

y_pred = model.predict_classes(x_test)
y_act = Y[~msk]
M = confusion_matrix(y_pred, y_act)

pdts
print(M)

print('Classification Report')
target_names = pdts
print(classification_report(y_act, y_pred, target_names=target_names))
#matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))


model.save("model.h5")





