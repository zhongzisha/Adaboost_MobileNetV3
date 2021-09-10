

from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16, MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class_names = ('nontower', 'normal', 'jieduan', 'wanzhe')
# new_input = Input(shape=(640, 480, 3))
# model = VGG16(weights=None, input_tensor=new_input)
base_model = MobileNetV3Small(weights=None, include_top=False)
x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
x = Reshape(target_shape=(-1, 1024))(x)
predictions = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy', 'accuracy'])
model.summary()
# model = MobileNetV3Large(weights=None, input_tensor=new_input)
# model.summary()


train_data_root = '/share/home/zhongzisha/datasets/ganta_patch_classification/train'
val_data_root = '/share/home/zhongzisha/datasets/ganta_patch_classification/val'
train_ds = image_dataset_from_directory(directory=train_data_root, class_names=class_names)
val_ds = image_dataset_from_directory(directory=val_data_root, class_names=class_names)
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))
model.fit(train_ds,
          epochs=50,
          verbose=True,
          validation_data=val_ds)












