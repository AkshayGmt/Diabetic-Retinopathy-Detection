import numpy as np # linear algebra
import pandas as pd

# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
#from keras.layers.normalization import BatchNormalization
from keras.layers import BatchNormalization


# Initializing the CNN
classifier = Sequential()

# Convolution Step 1
classifier.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(224, 224, 3), activation = 'relu'))

# Max Pooling Step 1
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
classifier.add(BatchNormalization())

# Convolution Step 2
classifier.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))

# Max Pooling Step 2

classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))
classifier.add(BatchNormalization())

# Convolution Step 3
classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
classifier.add(BatchNormalization())

# Convolution Step 4
classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
classifier.add(BatchNormalization())

# Convolution Step 5
classifier.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))

# Max Pooling Step 3
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
classifier.add(BatchNormalization())

# Flattening Step
classifier.add(Flatten())

# Full Connection Step
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dropout(0.6))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 1000, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 5, activation = 'softmax'))
classifier.summary()



# let's visualize layer names and layer indices to see how many layers
# we should freeze:
from keras import layers
for i, layer in enumerate(classifier.layers):
   print(i, layer.name)
   print("layerssssssss")


# we chose to train the top 2 conv blocks, i.e. we will freeze
# the first 8 layers and unfreeze the rest:
print("Freezed layers:")
for i, layer in enumerate(classifier.layers[:20]):
    print(i, layer.name)
    print("layyyyyyyesse2")
    layer.trainable = False


#trainable parameters decrease after freezing some bottom layers   
classifier.summary()


#from keras import optimizers
from tensorflow.keras.optimizers import SGD
classifier.compile(optimizer=SGD(lr=0.001, momentum=0.5, decay=0.005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# image preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.4,
                                   zoom_range=0.4,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 128
#base_dir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"

training_set = train_datagen.flow_from_directory('training/',
                                                 target_size=(224, 224),
                                                 batch_size=batch_size,
                                                 class_mode='categorical')

valid_set = valid_datagen.flow_from_directory('validation/',
                                            target_size=(224, 224),
                                            batch_size=batch_size,
                                            class_mode='categorical')

class_dict = training_set.class_indices
print(class_dict)

li = list(class_dict.keys())
print(li)

train_num = training_set.samples
valid_num = valid_set.samples
# checkpoint
from keras.callbacks import ModelCheckpoint
weightpath = "best_weights_9.hdf5"
checkpoint = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [checkpoint]

#fitting images to CNN
history = classifier.fit_generator(training_set,
                         steps_per_epoch=train_num//batch_size,
                         validation_data=valid_set,
                         epochs=30,
                         validation_steps=valid_num//batch_size,
                         callbacks=callbacks_list)
#saving model
filepath="AlexNetModel1.hdf5"
classifier.save(filepath)
classifier.load_weights('AlexNetModel1.hdf5')

print(history.history.keys())

#plotting training values
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

#print(acc)

# predicting an image
from keras.preprocessing import image
import numpy as np
image_path = "output_image1.jpg"
new_img = image.load_img(image_path, target_size=(224, 224))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
img = img/255

#print("Following is our prediction:")
prediction = classifier.predict(img)
#print(prediction)

prediction = np.array(prediction )
max_value = max(prediction[0])
max_positions = [i for i, x in enumerate(prediction[0]) if x == max_value]
#print("Maximum value:", max_value)
#print("Positions of maximum value:", max_positions)
my_dict = {"0 - No_DR": 0, "1 - Mild": 1, "2 - Moderate": 2, " 3 - Severe":3, " 4 - Proliferate_DR":4}
max_positions=np.array(max_positions)
keys = [key for key, value in my_dict.items() if value == max_positions]
print(f"The image  DR is/are: {keys}")

# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# d = prediction.flatten()
# j = d.max()
# for index,item in enumerate(d):
#     if item == j:
#         class_name = li[index]

# ##Another way
# img_class = classifier.predict(img)
# img_prob = classifier.predict_proba(img)
# print(img_class ,img_prob )



# print(class_name)
# #ploting image with predicted class name        
# plt.figure(figsize = (4,4))
# plt.imshow(new_img)
# plt.axis('off')
# plt.title(class_name)
# plt.show()





