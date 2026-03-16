import cv2

net = cv2.dnn.readNetFromCaffe('MobileNet/mobilenet_deploy.prototxt', 'MobileNet/mobilenet.caffemodel')
classes = []
table = []
#----------------------------------------1----------------------------------------------
with open('MobileNet/synset.txt', 'r', encoding = 'utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split(' ', 1)
        name = parts[1] if len(parts) > 1 else[0]
        classes.append(name)


image = cv2.imread('images/Mobilenet/11.jpg')
image = cv2.resize(image, (1000, 600))
blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5))#zobrazhenya rozmir mashtab piksiliv mahtabyvanya


net.setInput(blob)
preds = net.forward()

index = preds[0].argmax()

label = classes[index] if index < len(classes) else "unknown"
table.append(label)
conf = float(preds[0][index].item()) * 100

print(f'Клас: {label}')
print(f'Ймовірність: {round(conf, 2)}%')


text = label + ': ' + str(int(conf)) + '%'
cv2.putText(image, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#---------------------------------------------------2---------------------------------------------------------
clases = []

with open('MobileNet/synset.txt', 'r', encoding = 'utf-8') as f:
    for line1 in f:
        line1 = line1.strip()
        if not line1:
            continue

        parts1 = line1.split(' ', 1)
        name1 = parts[1] if len(parts1) > 1 else[0]
        clases.append(name1)


image1 = cv2.imread('images/Mobilenet/12.jpg')
image1 = cv2.resize(image1, (1000, 600))
blob1 = cv2.dnn.blobFromImage(cv2.resize(image1, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5))#zobrazhenya rozmir mashtab piksiliv mahtabyvanya


net.setInput(blob1)
preds1 = net.forward()

index1 = preds1[0].argmax()

label1 = clases[index1] if index1 < len(clases) else "unknown"
table.append(label1)
conf1 = float(preds1[0][index1].item()) * 100

print(f'Клас: {label1}')
print(f'Ймовірність: {round(conf1, 2)}%')


text1 = label1 + ': ' + str(int(conf)) + '%'
cv2.putText(image1, text1, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#-----------------------------------------------3------------------------------------------------

clases1 = []

with open('MobileNet/synset.txt', 'r', encoding = 'utf-8') as f:
    for line2 in f:
        line2 = line2.strip()
        if not line2:
            continue

        parts2 = line2.split(' ', 1)
        name2 = parts2[1] if len(parts2) > 1 else[0]
        clases1.append(name2)


image2 = cv2.imread('images/Mobilenet/13.jpg')
image2 = cv2.resize(image2, (1000, 600))
blob2 = cv2.dnn.blobFromImage(cv2.resize(image2, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5))#zobrazhenya rozmir mashtab piksiliv mahtabyvanya


net.setInput(blob2)
preds2 = net.forward()

index2 = preds2[0].argmax()

label2 = clases1[index2] if index2 < len(clases1) else "unknown"
table.append(label2)
conf2 = float(preds2[0][index2].item()) * 100

print(f'Клас: {label2}')
print(f'Ймовірність: {round(conf2, 2)}%')


text2 = label2 + ': ' + str(int(conf2)) + '%'
cv2.putText(image2, text2, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#-----------------------------------------------------4------------------------------------------------
clases2 = []

with open('MobileNet/synset.txt', 'r', encoding = 'utf-8') as f:
    for line3 in f:
        line3 = line3.strip()
        if not line3:
            continue

        parts3 = line3.split(' ', 1)
        name3 = parts3[1] if len(parts3) > 1 else[0]
        clases2.append(name3)


image3 = cv2.imread('images/Mobilenet/14.jpg')
image3 = cv2.resize(image3, (1000, 600))
blob3 = cv2.dnn.blobFromImage(cv2.resize(image3, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5))#zobrazhenya rozmir mashtab piksiliv mahtabyvanya


net.setInput(blob3)
preds3 = net.forward()

index3 = preds3[0].argmax()

label3 = clases2[index3] if index3 < len(clases2) else "unknown"
table.append(label3)
conf3 = float(preds3[0][index3].item()) * 100

print(f'Клас: {label3}')
print(f'Ймовірність: {round(conf3, 2)}%')

text3 = label3 + ': ' + str(int(conf3)) + '%'
cv2.putText(image3, text3, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)




cv2.imshow('Result 4', image3)
cv2.waitKey(0)
cv2.imshow('Result 3', image2)
cv2.waitKey(0)
cv2.imshow('Result 2', image1)
cv2.waitKey(0)
cv2.imshow('Result @_@', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Таблиця:")

checked = []

for a in range(len(table)):
    if table[a] not in checked:
        count = 0

        for j in range(len(table)):
            if table[a] == table[j]:
                count += 1

        print(table[a], ":", count)
        checked.append(table[a])


