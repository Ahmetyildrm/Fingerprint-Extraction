import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("Images/SampleFinger.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, channels = image.shape
print("Image Size: ", height, width)


alpha = 0.8  # Contrast control (1.0-3.0)
beta = 25  # Brightness control (0-100)

adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

grayscale = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY)

HistEq = cv2.equalizeHist(grayscale)

norm_img = np.zeros((800, 800))
NormalizedImage = cv2.normalize(HistEq,  norm_img, 0, 255, cv2.NORM_MINMAX)

BlocksizeC = int(height*width/9792)
if BlocksizeC %2 ==0 : BlocksizeC = BlocksizeC+1

GThreshold = cv2.adaptiveThreshold(NormalizedImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, BlocksizeC, 2)

kernel = np.ones((5,5),np.float32)/25
smoothed = cv2.filter2D(GThreshold,-1,kernel) # Sonuç istediğin çıkmazsa buradaki smoothing yöntemini ya da parametreleri değiştirebilirsin

ikinciadj = cv2.convertScaleAbs(smoothed, alpha=1.7, beta=-40)

Gblur = cv2.GaussianBlur(ikinciadj, (5, 5), 0)

weighted = Gblur

_,BThreshold = cv2.threshold(weighted, 200, 255, cv2.THRESH_BINARY)

CroppedFinal = BThreshold[int(height/4):int(height/4)+int(height/3), int(width/4):int(width/4)+int(height/3)]


titles = ["Image", "Adjusted", "Grayscale", "Histogram Equalization", "Normalized", "Gaussian Threshold", "Smoothed", "İkinci Adjustment", "Gaussian Blur", "Weighted", "Binary Threshold", "Cropped Part"]
images = [image, adjusted, grayscale, HistEq, NormalizedImage, GThreshold, smoothed, ikinciadj, Gblur, weighted, BThreshold, CroppedFinal]


for i in range(12):
    plt.subplot(3, 4, i + 1), plt.imshow(images[i], "gray")
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
