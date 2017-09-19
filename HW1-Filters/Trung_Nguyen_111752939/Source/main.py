# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np


def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Histogram equalization")
    print("2 Frequency domain filtering")
    print("3 Laplacian pyramid blending")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image] " +
          "[output directory]")  # Single input, single output
    print(sys.argv[0] + " 2 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, three outputs
    print(sys.argv[0] + " 3 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, single output


# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================


def histogram_equalization(img_in):

    img = img_in
    img2 = img_in
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])

        # Get the cumulative distribution function
        cdf = histr.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)

        # Normalization so that the range becomes [0 255]
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')

        # Mapping function: the intensity values of the original
        # in the original image is the index of the cdf
        img2[:, :, i] = cdf[img[:, :, i]]
        img_out = img2

    return True, img_out


def Question1():

    # Read in input images
    input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    # Histogram equalization
    succeed, output_image = histogram_equalization(input_image)

    # Write out the result
    output_name = sys.argv[3] + "1.jpg"
    cv2.imwrite(output_name, output_image)

    return True


# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================


def low_pass_filter(img_in):

    # I don't want to change the structure of this main file
    # so I write ft() and ift() inside each method (3 times in total)
    def ft(img, newsize=None):
        dft = np.fft.fft2(np.float32(img), newsize)
        return np.fft.fftshift(dft)

    def ift(shift):
        f_ishift = np.fft.ifftshift(shift)
        img_back = np.fft.ifft2(f_ishift)
        return np.abs(img_back)

    # Convert original image to grayscale
    img = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)

    # FFT of input
    dft = ft(img)

    # Create a mask of 0 of same size with input image
    rows, cols = img.shape
    crow, ccol = rows / 2, cols / 2
    mask = np.zeros((rows, cols), np.uint8)

    # Mask center size is 20x20 (10 each side from the center)
    mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 1

    # Multiplication in the frequency domain
    fshift = dft * mask

    # IFFT and Convert to uint8 to write image later
    img_out = np.uint8(ift(fshift))
    return True, img_out


def high_pass_filter(img_in):

    def ft(img, newsize=None):
        dft = np.fft.fft2(np.float32(img), newsize)
        return np.fft.fftshift(dft)

    def ift(shift):
        f_ishift = np.fft.ifftshift(shift)
        img_back = np.fft.ifft2(f_ishift)
        return np.abs(img_back)

    # Convert original image to grayscale
    img = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)

    # FFT of input
    dft = ft(img)

    # Create a mask of 1 of the same size with input image
    rows, cols = img.shape
    crow, ccol = rows / 2, cols / 2
    mask = np.ones((rows, cols), np.uint8)

    # Mask center size is 20x20 (10 each side from the center)
    mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 0
    fshift = dft * mask
    img_out = np.uint8(ift(fshift))

    return True, img_out


def deconvolution(img_in):

    img = img_in

    # Create the Gaussian Kernel of size 21 and sigma = 5
    gk = cv2.getGaussianKernel(21, 5)
    gk = gk * gk.T

    def ft(img, newsize=None):
        dft = np.fft.fft2(np.float32(img), newsize)
        return np.fft.fftshift(dft)

    def ift(shift):
        f_ishift = np.fft.ifftshift(shift)
        img_back = np.fft.ifft2(f_ishift)
        return np.abs(img_back)

    dft = ft(img, (img.shape[0], img.shape[1]))  # make sure sizes match
    gkf = ft(gk, (img.shape[0], img.shape[1]))  # so we can multiple easily

    # Division in the frequency domain
    fshift = dft / gkf

    # Multiply by 255 becauce the input is in range [0 1] instead of [0 255]
    img_out = np.uint8(255 * (ift(fshift)))

    return True, img_out


def Question2():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3],
                              cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Low and high pass filter
    succeed1, output_image1 = low_pass_filter(input_image1)
    succeed2, output_image2 = high_pass_filter(input_image1)

    # Deconvolution
    succeed3, output_image3 = deconvolution(input_image2)

    # Write out the result
    output_name1 = sys.argv[4] + "2.jpg"
    output_name2 = sys.argv[4] + "3.jpg"
    output_name3 = sys.argv[4] + "4.jpg"
    cv2.imwrite(output_name1, output_image1)
    cv2.imwrite(output_name2, output_image2)
    cv2.imwrite(output_name3, output_image3)

    return True


# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================


def laplacian_pyramid_blending(img_in1, img_in2):

    A = img_in1
    B = img_in2

    # Make the two images square
    A = A[:, :A.shape[0]]
    B = B[:A.shape[0], :A.shape[0]]

    # Generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpA.append(G)

    # Generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpB.append(G)

    # Generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)

    # Generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)

    # Add left and right halves of the low frequency componets in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
        LS.append(ls)

    # Add the high frequency components in each level
    ls_ = LS[0]
    for i in xrange(1, 6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    img_out = ls_

    return True, img_out


def Question3():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR)

    # Laplacian pyramid blending
    succeed, output_image = laplacian_pyramid_blending(input_image1,
                                                       input_image2)

    # Write out the result
    output_name = sys.argv[4] + "5.jpg"
    cv2.imwrite(output_name, output_image)

    return True


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) < 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])

        if (question_number == 1 and not (len(sys.argv) == 4)):
            help_message()
            sys.exit()
        if (question_number == 2 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number == 3 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
            print("Input parameters out of bound ...")
            sys.exit()

    function_launch = {
        1: Question1,
        2: Question2,
        3: Question3,
    }

    # Call the function
    function_launch[question_number]()
