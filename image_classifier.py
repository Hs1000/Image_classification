import numpy as np
import pandas as pd
import tensorflow as tf
model=tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')
])

model.summary()

from tensorflow.keras.optimizers import SGD

model.compile(loss='binary_crossentropy',
               optimizer=SGD(lr=0.01),
               metrics=['acc'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1/255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1/255)

train_generator=train_datagen.flow_from_directory(
        r"/Users/hardiksingh/Desktop/Image_classifier/dataset/training_set",
        target_size=(150,150),
        batch_size=80,
        class_mode='binary')


validation_generator=train_datagen.flow_from_directory(
        r"/Users/hardiksingh/Desktop/Image_classifier/dataset/test_set",
        target_size=(150,150),
        batch_size=80,
        class_mode='binary')

history=model.fit_generator(train_generator,
                            steps_per_epoch=100,
                            epochs=30,
                            validation_data=validation_generator,
                            validation_steps=25,
                            verbose=2)


from keras.preprocessing import image
test_image=image.load_img('cat_or_dog_2.jpg',target_size=(150,150))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=model.predict(test_image)

if result[0][0] > 0.5:
    prediction='dog'
else:
    prediction='cat'
print(prediction)
