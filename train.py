import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to save the model after the training")
ap.add_argument("-i", "--imagenet", type=str, default="imagenet", help="path to pretained imagenet for MobileNetV2 or it will be downloaded")
ap.add_argument("-tr", "--train", type=str, required=True, help="path to the training dataset folder")
ap.add_argument("-ts", "--test", type=str, required=True, help="path to the testing dataset folder")
ap.add_argument("-trs", "--train-size", type=int, default=400, help="number of the training dataset")
ap.add_argument("-tss", "--test-size", type=int, default=100, help="number of the testing dataset")
args = vars(ap.parse_args())

import numpy as np
from keras.preprocessing import image
from keras import applications
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, GlobalAveragePooling2D


classes = [1, 2, 3, 4, 5]

base_model = applications.MobileNetV2(weights=args["imagenet"], include_top=False,)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False


traingen = image.ImageDataGenerator(rescale=1/255, 
                                    horizontal_flip=True, 
                                    vertical_flip=True,
                                    rotation_range=180,
                                   zoom_range=[1, 0.5],)
train_data = traingen.flow_from_directory(args["train"], 
	target_size=(224,224), 
	batch_size=32)

testgen = image.ImageDataGenerator(rescale=1/255)
test_data = traingen.flow_from_directory(args["test"], 
	target_size=(224,224), 
	batch_size=16)


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

H = model.fit_generator(train_data, epochs=10, 
                        steps_per_epoch=int(400//32), 
                        validation_data=test_data, 
                        validation_steps=int(100//16),)


for layer in model.layers[:80]:
    layer.trainable = False

for layer in model.layers[80:]:
    layer.trainable = True


from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_data, epochs=5, 
                    steps_per_epoch=int(args["train-size"]//32),
                   validation_data=test_data,
                    validation_steps=int(args["test-size"]//16),)

model.save(args["model"])
