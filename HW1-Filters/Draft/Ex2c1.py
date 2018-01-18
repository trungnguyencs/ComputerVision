import cv2
import numpy as np

img = cv2.imread("blurred2.png", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

gk = cv2.getGaussianKernel(21,5)
gk = gk * gk.T

def ft(img, newsize=None):
    dft = np.fft.fft2(np.float32(img),newsize)
    return np.fft.fftshift(dft)

def ift(shift):
    f_ishift = np.fft.ifftshift(shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

dft = ft(img, (img.shape[0],img.shape[1])) # make sure sizes match
gkf = ft(gk, (img.shape[0],img.shape[1])) # so we can multiple easily

fshift = dft/gkf
img_back = np.uint8(255*(ift(fshift)))
print img_back.shape
# cv2.imshow('a',img_back)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)
cv2.imwrite('b.png',img_back)