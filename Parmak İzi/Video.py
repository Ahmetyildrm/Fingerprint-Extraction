import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter('camera.avi',fourcc, 30.0, (640,480), 0) # Sondaki 0 grayscale kaydedeceğimiz için


height = 640
width = 480
print(height, width)

while True:
    _, frame = cap.read()

    cv2.imshow("Input", frame)
    alpha = 0.8 # Contrast control (1.0-3.0)
    beta = 25 # Brightness control (0-100)

    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    grayscale = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY)

    HistEq = cv2.equalizeHist(grayscale)

    norm_img = np.zeros((800, 800))
    NormalizedImage = cv2.normalize(HistEq,  norm_img, 0, 255, cv2.NORM_MINMAX)

    BlocksizeC = int(height * width / 9792)
    if BlocksizeC % 2 == 0: BlocksizeC = BlocksizeC + 1

    GThreshold = cv2.adaptiveThreshold(NormalizedImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, BlocksizeC, 2)

    kernel = np.ones((5,5),np.float32)/25
    smoothed = cv2.filter2D(GThreshold,-1,kernel) # Sonuç istediğin çıkmazsa buradaki smoothing yöntemini ya da parametreleri değiştirebilirsin

    ikinciadj = cv2.convertScaleAbs(smoothed, alpha=1.7, beta=-40)

    Gblur = cv2.GaussianBlur(ikinciadj, (5, 5), 0)

    weighted = Gblur

    _,BThreshold = cv2.threshold(weighted, 200, 255, cv2.THRESH_BINARY)

    CroppedFinal = BThreshold[int(height/4):int(height/4)+int(height/3), int(width/4):int(width/4)+int(height/3)]

    out.write(BThreshold)  # Frameleri çıktı almak
    cv2.imshow("Input", BThreshold)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()