import tensorflow as tf
import DataLoader



def getModel():
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Rescaling(1. / 255, input_shape=(DataLoader.WIDTH, DataLoader.HEIGHT, 3)),
    #     tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(6, activation="softmax")
    # ])
    #
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.CategoricalCrossentropy(),
    #               metrics=['accuracy'],
    #               )

    final_cnn = tf.keras.Sequential()

    final_cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                         input_shape=(200, 200, 3)))  # 3rd dim = 1 for grayscale images.
    final_cnn.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))

    final_cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    final_cnn.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))

    final_cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
    final_cnn.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))

    final_cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
    final_cnn.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))

    final_cnn.add(tf.keras.layers.GlobalAveragePooling2D())

    final_cnn.add(tf.keras.layers.Dense(132, activation='relu'))

    final_cnn.add(tf.keras.layers.Dense(6, activation='softmax'))

    final_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    # final_cnn.summary()

    return final_cnn
