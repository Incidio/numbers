
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
import numpy as np
from io import BytesIO
from PIL import Image

INPUT_SHAPE = 784  # Плоский вектор для MNIST
CLASS_COUNT = 10

# Инициализация модели (та же архитектура)
model = Sequential()
model.add(Dense(800, input_dim=INPUT_SHAPE, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(CLASS_COUNT, activation='softmax'))

# Загрузка весов
model.load_weights('model.weights.h5')

def process(image_file):
    # Открытие изображения
    image = Image.open(BytesIO(image_file)).convert('L')  # Grayscale
    # Resize to 28x28
    resized_image = image.resize((28, 28))
    # Преобразование в массив, flatten, normalize
    array = np.array(resized_image).reshape(1, -1).astype('float32') / 255.
    # Предсказание
    prediction_array = model.predict(array)[0]
    # Возврат как строка списка вероятностей
    return str(list(prediction_array))
