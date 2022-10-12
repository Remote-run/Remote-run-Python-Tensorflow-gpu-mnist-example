import os

import tensorflow as tf

from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)



# setup the save stuff
checkpoint_path = "./save_data/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print(checkpoint_dir)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)




mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
y_test = tf.cast(y_test, tf.float32)

dropout = 0.75 # Dropout, probability to keep units
learning_rate = 0.001


model = tf.keras.models.Sequential([
    layers.Reshape(target_shape=[ 28, 28,1]),
    layers.Conv2D( 64, 5, activation=tf.nn.relu),
    layers.MaxPooling2D(( 2, 2)),
    layers.Conv2D( 256, 3, activation=tf.nn.relu),
    layers.Conv2D( 512, 3, activation=tf.nn.relu),
    layers.MaxPooling2D(( 2, 2)),
    layers.Flatten(),
    layers.Dense( 1024),
    layers.Dropout(dropout),
    layers.Dense( 10),
])


optimizers = optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizers,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=3, 
                    validation_data=(x_test, y_test),callbacks=[cp_callback])


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.savefig('./save_data/acc_plot.png')




test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print("test loss", test_loss)
print("test acc", test_acc)
