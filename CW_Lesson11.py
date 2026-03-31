import pandas as pd # для роботи з csv
import numpy as np # математичні операції
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras #ля айпішек тенсорфлор
from tensorflow.keras import layers # для шарів
from sklearn.preprocessing import LabelEncoder# перетворює текстові мітки в числа
import matplotlib.pyplot as plt

df = pd.read_csv('data/figures.csv')
print(df.head())

encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])
X = df[['area','perimeter', 'corners']]
y = df['label_enc']

model = keras.Sequential([layers.Dense(8, activation='relu', input_shape = (3, )),
                          layers.Dense(8, activation='relu'),
                          layers.Dense(3, activation='softmax')
                          ])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(X, y, epochs = 200, verbose = 0)

plt.plot(history.history['loss'], label = 'Втрати')
plt.plot(history.history['accuracy'], label = 'Точність')
plt.xlabel('Епоха')
plt.ylabel('Значення')
plt.title('Процес навчання моделі')
plt.legend()
plt.show()


test = np.array([16,16,0])

pred = model.predict(test)
print(f'Ймовірність кожного класу {pred}')
print(f'Результат {encoder.inverse_transform(np.argmax(pred))}')