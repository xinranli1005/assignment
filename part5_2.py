
import numpy as np
import matplotlib.pyplot as plt

import datetime
import Umodels

starttime = datetime.datetime.now()

new_train = np.load('../../database/CatDog/new_train.npy')
new_valid = np.load('../../database/CatDog/new_valid.npy')
new_test = np.load('../../database/CatDog/new_test.npy')
blur_train = np.load('../../database/CatDog/blur_train.npy')
blur_valid = np.load('../../database/CatDog/blur_valid.npy')
blur_test = np.load('../../database/CatDog/blur_test.npy')

# De-Noising a Gaussian Blurred Image
# The aim is to de-noise a blurred given image using Convolutional Neural Networks.
# https://github.com/done-n-dusted/deblur-fashionmnist/blob/master/Denoising_v1.ipynb
#Plotting few images from the dataset



#importing required packages for model building

input_size = (128,128,3)
#
model = Umodels.model_2s(input_size)  #22057 parameters
model.summary()


from keras import optimizers
opt = optimizers.SGD(lr=1e-4)

model.compile(loss = 'mse', optimizer = 'adam', metrics =['mse'])


history=model.fit(blur_train.reshape(-1, 128, 128, 3), new_train.reshape(-1, 128, 128, 3), epochs = 2, batch_size = 2000,
              validation_data = (blur_valid.reshape(-1, 128, 128, 3), new_valid.reshape(-1, 128, 128, 3)))
#history=model.fit(blur_train.reshape(-1, 128, 128, 3), new_train.reshape(-1, 128, 128, 3), epochs = 60, batch_size = 50,
              #validation_data = (blur_valid.reshape(-1, 128, 128, 3), new_valid.reshape(-1, 128, 128, 3)))

#show mse
plt.plot(history.history['mse'],'r',label='Training MSE')
plt.plot(history.history['val_mse'],'b',label='Validation MSE')
plt.title('Part3-Training and Validation Curve of MSE')
plt.ylabel('MSE')
plt.xlabel('epochs')
plt.legend(['Training MSE', 'Validation MSE'], loc='upper right')
plt.show()

#predict
ANN_preds = model.predict(blur_test.reshape(-1, 128, 128, 3))
ANN_preds = ANN_preds.reshape(-1, 128, 128)

def mse2(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true), axis=-1)
print('MSE(testing)'+str(mse2(new_test.reshape(-1,6553600),ANN_preds.reshape(-1,6553600))))

# Testing the model

# randomly sort test
indexes = np.arange(new_test.shape[0])
for _ in range(5): indexes = np.random.permutation(indexes)  # shuffle 5 times!
new_test = new_test[indexes]
blur_test = blur_test[indexes]
ANN_preds = ANN_preds[indexes]

# randomly select
new_test = new_test[:10]
blur_test = blur_test[:10]
ANN_preds = ANN_preds[:10]
num = 10

plt.figure(figsize = (15, 15))
#print('Original Images')
for i in range(num):
    plt.subplot(5, num, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Original')
    plt.imshow(new_test[i], cmap=plt.cm.binary)

    plt.subplot(5, num, i + 11)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Blur')
    plt.imshow(blur_test[i], cmap=plt.cm.binary)

    plt.subplot(5, num, i + 21)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('ANN_Preds')
    plt.imshow(ANN_preds[i], cmap=plt.cm.binary)

    new_test_1 = new_test[i]
    blur_test_1 = blur_test[i]
    plt.subplot(5, num, i + 31)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Org_Blur')
    plt.xlabel(str(mse2(new_test_1.reshape(-1, 16384), blur_test_1.reshape(-1, 16384))))
    plt.imshow((ANN_preds[i] - new_test[i]), cmap=plt.cm.binary)

    new_test_1 = new_test[i]
    ANN_preds_1 = ANN_preds[i]
    plt.subplot(5, num, i + 41)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Org_Preds')
    plt.xlabel(str(mse2(new_test_1.reshape(-1, 16384), ANN_preds_1.reshape(-1, 16384))))
    plt.imshow((ANN_preds[i] - new_test[i]), cmap=plt.cm.binary)

plt.show()

endtime = datetime.datetime.now()
time=endtime - starttime
print ("training_time=" + str(time))
