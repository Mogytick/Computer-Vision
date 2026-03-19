import cv2
import numpy as np
from pyexpat import features
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def generate_image(color, shape):
    img = np.zeros((200,200,3), np.uint8)

    if shape == 'circle':
        cv2.circle(img, (100,100), 50, color, -1)
    elif shape == 'rectangle':
        cv2.rectangle(img, (50,50), (150,150), color, -1)
    elif shape == 'triangle':
        points = np.array([[100,40],[40,160],[160,100]])
        cv2.drawContours(img, [points], 0, color, -1)
    return img

x = []#spisok oznak-xarakteristiki
y = []#spisok mitok-nazva elementy

colors = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'cyan': (255, 255, 0),
    'magenta': (255, 0, 255),
    'orange': (0, 165, 255),
    'white': (255, 255, 255)
}

shapes = ['circle', 'rectangle', 'triangle']

for color_name, bgr in colors.items():
    for shape in shapes:
        for _ in range(10):
            img = generate_image(bgr, shape)
            mean_color = cv2.mean(img)[:3]
            features = [mean_color[0], mean_color[1], mean_color[2]]

            x.append(features)
            y.append(f'{color_name}_{shape}')

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, stratify = y)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)

accuracy = model.score(x_test,y_test)
print(f'Точність моделі: {round(accuracy * 100, 2)}%')
colors_list = []

for i in range(5):
    test_img = generate_image((255, 255, 0), 'triangle')
    mean_color = cv2.mean(test_img)[:3]
    colors_list.append(mean_color)

avg_color = np.mean(colors_list, axis=0)
features = model.predict([avg_color])
print(f'Передбачення: {features}')
cv2.imshow('test', test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()