import cv2
import matplotlib.pyplot as plt

image = cv2.imread("jurassic-park-tour-jeep.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
r = 50.0 / image.shape[1]
dim = (100, int(image.shape[0] * r))

# perform the actual resizing of the image and show it
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
print image.shape
print resized.shape

plt.axis("off")
plt.imshow(resized)
plt.show()
