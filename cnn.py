
import tensorflow 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from tensorflow.keras.preprocessing import image




train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    r'Python\dataset\train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    r'Python\dataset\test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)


cnn = Sequential()


cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))


cnn.add(MaxPooling2D(pool_size=2, strides=2))


cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, strides=2))


cnn.add(Flatten())


cnn.add(Dense(units=128, activation='relu'))


cnn.add(Dense(units=23, activation='softmax'))


checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)


cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


cnn.fit(x=training_set, validation_data=test_set, epochs=25, callbacks=[checkpoint, early_stop])


def make_prediction(image_path):
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image /= 255.0
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)
    return result


result = make_prediction(r'Python\dataset\test\Hair Loss Photos Alopecia and other Hair Diseases\acne-keloidalis-18.jpg')
print(result)




