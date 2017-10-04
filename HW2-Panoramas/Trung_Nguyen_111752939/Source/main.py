# Instructions:
# Do not change the output file names, use the helper functions as you see fit

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.ma as ma


def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Perspective warping")
    print("2 Cylindrical warping")
    print("3 Bonus perspective warping")
    print("4 Bonus cylindrical warping")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image1] " +
          "[path to input image2] " + "[path to input image3] " +
          "[output directory]")


'''
Detect, extract and match features between img1 and img2.
Using SIFT as the detector/extractor, but this is inconsequential to the user.

Returns: (pts1, pts2), where ptsN are points on image N.
    The lists are "aligned", i.e. point i in pts1 matches with point i in pts2.

Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (pts1, pts2) = feature_matching(im1, im2)

    plt.subplot(121)
    plt.imshow(im1)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
    plt.subplot(122)
    plt.imshow(im2)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
'''


def feature_matching(img1, img2, savefig=False):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches2to1 = flann.knnMatch(des2, des1, k=2)

    matchesMask_ratio = [[0, 0] for i in xrange(len(matches2to1))]
    match_dict = {}
    for i, (m, n) in enumerate(matches2to1):
        if m.distance < 0.7 * n.distance:
            matchesMask_ratio[i] = [1, 0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1, des2, k=2)
    matchesMask_ratio_recip = [[0, 0] for i in xrange(len(recip_matches))]

    for i, (m, n) in enumerate(recip_matches):
        if m.distance < 0.7 * n.distance:  # ratio
            if m.queryIdx in match_dict and match_dict[m.
                                                       queryIdx] == m.trainIdx:  #reciprocal
                good.append(m)
                matchesMask_ratio_recip[i] = [1, 0]

    if savefig:
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=matchesMask_ratio_recip,
            flags=0)
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, recip_matches, None,
                                  **draw_params)

        plt.figure(), plt.xticks([]), plt.yticks([])
        plt.imshow(img3,)
        plt.savefig("feature_matching.png", bbox_inches='tight')

    return ([kp1[m.queryIdx].pt
             for m in good], [kp2[m.trainIdx].pt for m in good])


'''
Warp an image from cartesian coordinates (x, y) into cylindrical coordinates (theta, h)
Returns: (image, mask)
Mask is [0,255], and has 255s wherever the cylindrical images has a valid value.
Masks are useful for stitching

Usage example:

    im = cv2.imread("myimage.jpg",0) #grayscale
    h,w = im.shape
    f = 700
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    imcyl = cylindricalWarpImage(im, K)
'''


def cylindricalWarpImage(img1, K, savefig=False):
    f = K[0, 0]

    im_h, im_w = img1.shape

    # go inverse from cylindrical coord to the image
    # (this way there are no gaps)
    cyl = np.zeros_like(img1)
    cyl_mask = np.zeros_like(img1)
    cyl_h, cyl_w = cyl.shape
    x_c = float(cyl_w) / 2.0
    y_c = float(cyl_h) / 2.0
    for x_cyl in np.arange(0, cyl_w):
        for y_cyl in np.arange(0, cyl_h):
            theta = (x_cyl - x_c) / f
            h = (y_cyl - y_c) / f

            X = np.array([math.sin(theta), h, math.cos(theta)])
            X = np.dot(K, X)
            x_im = X[0] / X[2]
            if x_im < 0 or x_im >= im_w:
                continue

            y_im = X[1] / X[2]
            if y_im < 0 or y_im >= im_h:
                continue

            cyl[int(y_cyl), int(x_cyl)] = img1[int(y_im), int(x_im)]
            cyl_mask[int(y_cyl), int(x_cyl)] = 255

    if savefig:
        plt.imshow(cyl, cmap='gray')
        plt.savefig("cyl.png", bbox_inches='tight')

    return (cyl, cyl_mask)


'''
Calculate the geometric transform (only affine or homography) between two images,
based on feature matching and alignment with a robust estimator (RANSAC).

Returns: (M, pts1, pts2, mask)
Where: M    is the 3x3 transform matrix
       pts1 are the matched feature points in image 1
       pts2 are the matched feature points in image 2
       mask is a binary mask over the lists of points that selects the transformation inliers

Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (M, pts1, pts2, mask) = getTransform(im1, im2)

    # for example: transform im1 to im2's plane
    # first, make some room around im2
    im2 = cv2.copyMakeBorder(im2,200,200,500,500, cv2.BORDER_CONSTANT)
    # then transform im1 with the 3x3 transformation matrix
    out = cv2.warpPerspective(im1, M, (im1.shape[1],im2.shape[0]), dst=im2.copy(), borderMode=cv2.BORDER_TRANSPARENT)

    plt.imshow(out, cmap='gray')
    plt.show()
'''


def getTransform(src, dst, method='affine'):
    pts1, pts2 = feature_matching(src, dst)

    src_pts = np.float32(pts1).reshape(-1, 1, 2)
    dst_pts = np.float32(pts2).reshape(-1, 1, 2)

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(
            src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        M = np.append(M, [[0, 0, 1]], axis=0)

    if method == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    return (M, pts1, pts2, mask)


# ===================================================
# ================ Perspective Warping ==============
# ===================================================
def Perspective_warping(img1, img2, img3):

    im1 = img1
    im2 = img2
    im3 = img3

    im1 = cv2.copyMakeBorder(im1, 200, 200, 500, 500, cv2.BORDER_CONSTANT)
    (M1, pts2, pts1, mask) = getTransform(im2, im1, method='homography')
    out1 = cv2.warpPerspective(
        im2,
        M1, (im1.shape[1], im1.shape[0]),
        dst=im1.copy(),
        borderMode=cv2.BORDER_TRANSPARENT)

    (M2, pts2, pts1, mask) = getTransform(im3, out1, method='homography')
    out = cv2.warpPerspective(
        im3,
        M2, (im1.shape[1], im1.shape[0]),
        dst=out1.copy(),
        borderMode=cv2.BORDER_TRANSPARENT)

    output_image = out

    output_name = sys.argv[5] + "output_homography.png"
    cv2.imwrite(output_name, output_image)

    return True


def Bonus_perspective_warping(img1, img2, img3):

    im1 = img1
    im2 = img2
    im3 = img3

    def laplacian_pyramid_blending(img_in1, img_in2):

        # Write laplacian pyramid blending codes here
        A = img_in1
        B = img_in2
        deep_level = 5
        # generate Gaussian pyramid for A
        G = A.copy()
        gpA = [np.float32(G)]
        for i in xrange(0, deep_level + 1):
            G = cv2.pyrDown(G)
            gpA.append(np.float32(G))

        # generate Gaussian pyramid for B
        G = B.copy()
        gpB = [np.float32(G)]
        for i in xrange(0, deep_level + 1):
            G = cv2.pyrDown(G)
            gpB.append(np.float32(G))

        # generate Laplacian Pyramid for A, which contains high freq elements
        lpA = [gpA[deep_level]]
        for i in xrange(deep_level, 0, -1):
            GE = cv2.pyrUp(gpA[i])
            # subtraction requires same size matrix --> resize 1 image to have the same size if necessary
            height, width = np.shape(gpA[i - 1])[0], np.shape(gpA[i - 1])[1]
            if not np.shape(GE) == np.shape(gpA[i - 1]):
                L = cv2.subtract(gpA[i - 1], cv2.resize(GE, (width, height)))
            else:
                L = cv2.subtract(gpA[i - 1], GE)
            lpA.append(L)

        # generate Laplacian Pyramid for B
        lpB = [gpB[deep_level]]
        for i in xrange(deep_level, 0, -1):
            GE = cv2.pyrUp(gpB[i])
            # subtraction requires same size matrix --> resize 1 image to have the same size if necessary
            height, width = np.shape(gpB[i - 1])[0], np.shape(gpB[i - 1])[1]
            if not np.shape(GE) == np.shape(gpB[i - 1]):
                L = cv2.subtract(gpB[i - 1], cv2.resize(GE, (width, height)))
            else:
                L = cv2.subtract(gpB[i - 1], GE)
            lpB.append(L)

        # Now add left and right halves of images in each level
        LS = []
        for la, lb in zip(lpA, lpB):
            rows, cols = np.shape(la)
            ls = np.hstack((la[:, 0:np.int(cols / 2)],
                            lb[:, np.int(cols / 2):]))
            LS.append(ls)

        # now reconstruct
        ls_ = LS[0]  # LS[0] is the LOW freq element
        for i in xrange(1, deep_level + 1):
            ls_ = cv2.pyrUp(ls_)

            # addition requires same size matrix --> resize 1 image to have the same size if necessary
            height, width = np.shape(LS[i])[0], np.shape(LS[i])[1]
            if not np.shape(ls_) == np.shape(LS[i]):
                ls_ = cv2.add(cv2.resize(ls_, (width, height)), LS[i])
            else:
                ls_ = cv2.add(ls_, LS[i])

        # image with direct connecting each half
        real = np.hstack((A[:, :np.int(cols / 2)], B[:, np.int(cols / 2):]))

        return ls_

    im1 = cv2.imread("input1.png", 0)
    im2 = cv2.imread("input2.png", 0)
    im3 = cv2.imread("input3.png", 0)

    im1 = cv2.copyMakeBorder(im1, 200, 200, 500, 500, cv2.BORDER_CONSTANT)
    (M1, pts2, pts1, mask) = getTransform(im2, im1, method='homography')
    out1 = cv2.warpPerspective(
        im2,
        M1, (im1.shape[1], im1.shape[0]),
        dst=im1.copy(),
        borderMode=cv2.BORDER_TRANSPARENT)

    out_laplacian1 = laplacian_pyramid_blending(im1, out1)

    (M2, pts2, pts1, mask) = getTransform(im3, out1, method='homography')
    out2 = cv2.warpPerspective(
        im3,
        M2, (im1.shape[1], im1.shape[0]),
        dst=out1.copy(),
        borderMode=cv2.BORDER_TRANSPARENT)

    out_laplacian2 = laplacian_pyramid_blending(out2, out_laplacian1)

    output_image = out_laplacian2

    # Write out the result
    output_name = sys.argv[5] + "output_homography_lpb.png"
    cv2.imwrite(output_name, output_image)

    return True


# ===================================================
# =============== Cynlindrical Warping ==============
# ===================================================
def Cylindrical_warping(img1, img2, img3):

    im1 = img1
    im2 = img2
    im3 = img3

    h, w = im1.shape
    f = 450
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0,
                                                 1]])  # mock calibration matrix

    cyl1, cyl_mask1 = cylindricalWarpImage(im1, K)
    cyl1 = cv2.copyMakeBorder(cyl1, 50, 50, 300, 300, cv2.BORDER_CONSTANT)
    cyl2, cyl_mask2 = cylindricalWarpImage(im2, K)
    cyl3, cyl_mask3 = cylindricalWarpImage(im3, K)

    (M1, pts2, pts1, mask) = getTransform(cyl2, cyl1, method='affine')
    M1 = M1[:2]
    cyl2_extend = cv2.warpAffine(
        cyl2, M1, (cyl1.shape[1], cyl1.shape[0]), dst=cyl1.copy())
    mask_out2 = cv2.warpAffine(
        cyl_mask2, M1, (cyl1.shape[1], cyl1.shape[0]), dst=cyl1.copy() * 0)
    y = ma.array(cyl1, mask=mask_out2 / 255)
    out1 = cyl1.copy()
    out1[y.mask] = cyl2_extend[y.mask]

    (M2, pts2, pts1, mask) = getTransform(cyl3, cyl1, method='affine')
    M2 = M2[:2]
    cyl3_extend = cv2.warpAffine(
        cyl3, M2, (cyl1.shape[1], cyl1.shape[0]), dst=cyl1.copy())
    mask_out3 = cv2.warpAffine(
        cyl_mask3, M2, (cyl1.shape[1], cyl1.shape[0]), dst=cyl1.copy() * 0)
    y = ma.array(cyl1, mask=mask_out3 / 255)
    out2 = out1.copy()
    out2[y.mask] = cyl3_extend[y.mask]
    output_image = out2

    # Write out the result
    output_name = sys.argv[5] + "output_cylindrical.png"
    cv2.imwrite(output_name, output_image)

    return True


def Bonus_cylindrical_warping(img1, img2, img3):

    im1 = img1
    im2 = img2
    im3 = img3

    def laplacian_pyramid_blending(img_in1, img_in2):

        # Write laplacian pyramid blending codes here
        A = img_in1
        B = img_in2
        deep_level = 5
        # generate Gaussian pyramid for A
        G = A.copy()
        gpA = [np.float32(G)]
        for i in xrange(0, deep_level + 1):
            G = cv2.pyrDown(G)
            gpA.append(np.float32(G))

        # generate Gaussian pyramid for B
        G = B.copy()
        gpB = [np.float32(G)]
        for i in xrange(0, deep_level + 1):
            G = cv2.pyrDown(G)
            gpB.append(np.float32(G))

        # generate Laplacian Pyramid for A, which contains high freq elements
        lpA = [gpA[deep_level]]
        for i in xrange(deep_level, 0, -1):
            GE = cv2.pyrUp(gpA[i])
            # subtraction requires same size matrix --> resize 1 image to have the same size if necessary
            height, width = np.shape(gpA[i - 1])[0], np.shape(gpA[i - 1])[1]
            if not np.shape(GE) == np.shape(gpA[i - 1]):
                L = cv2.subtract(gpA[i - 1], cv2.resize(GE, (width, height)))
            else:
                L = cv2.subtract(gpA[i - 1], GE)
            lpA.append(L)

        # generate Laplacian Pyramid for B
        lpB = [gpB[deep_level]]
        for i in xrange(deep_level, 0, -1):
            GE = cv2.pyrUp(gpB[i])
            # subtraction requires same size matrix --> resize 1 image to have the same size if necessary
            height, width = np.shape(gpB[i - 1])[0], np.shape(gpB[i - 1])[1]
            if not np.shape(GE) == np.shape(gpB[i - 1]):
                L = cv2.subtract(gpB[i - 1], cv2.resize(GE, (width, height)))
            else:
                L = cv2.subtract(gpB[i - 1], GE)
            lpB.append(L)

        # Now add left and right halves of images in each level
        LS = []
        for la, lb in zip(lpA, lpB):
            rows, cols = np.shape(la)
            ls = np.hstack((la[:, 0:np.int(cols / 2)],
                            lb[:, np.int(cols / 2):]))
            LS.append(ls)

        # now reconstruct
        ls_ = LS[0]  # LS[0] is the LOW freq element
        for i in xrange(1, deep_level + 1):
            ls_ = cv2.pyrUp(ls_)

            # addition requires same size matrix --> resize 1 image to have the same size if necessary
            height, width = np.shape(LS[i])[0], np.shape(LS[i])[1]
            if not np.shape(ls_) == np.shape(LS[i]):
                ls_ = cv2.add(cv2.resize(ls_, (width, height)), LS[i])
            else:
                ls_ = cv2.add(ls_, LS[i])

        # image with direct connecting each half
        real = np.hstack((A[:, :np.int(cols / 2)], B[:, np.int(cols / 2):]))

        return ls_

    h, w = im1.shape
    f = 450
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0,
                                                 1]])  # mock calibration matrix

    cyl1, cyl_mask1 = cylindricalWarpImage(im1, K)
    cyl1 = cv2.copyMakeBorder(cyl1, 50, 50, 300, 300, cv2.BORDER_CONSTANT)
    cyl2, cyl_mask2 = cylindricalWarpImage(im2, K)
    cyl3, cyl_mask3 = cylindricalWarpImage(im3, K)

    (M1, pts2, pts1, mask) = getTransform(cyl2, cyl1, method='affine')
    M1 = M1[:2]
    out1 = cv2.warpAffine(
        cyl2,
        M1, (cyl1.shape[1], cyl1.shape[0]),
        dst=cyl1.copy() * 0,
        borderMode=cv2.BORDER_TRANSPARENT)

    out_laplacian1 = laplacian_pyramid_blending(cyl1, out1)

    (M2, pts2, pts1, mask) = getTransform(cyl3, out1, method='affine')
    M2 = M2[:2]
    out2 = cv2.warpAffine(
        cyl3,
        M2, (cyl1.shape[1], cyl1.shape[0]),
        dst=out1.copy() * 0,
        borderMode=cv2.BORDER_TRANSPARENT)

    out_laplacian2 = laplacian_pyramid_blending(out2, out_laplacian1)

    output_image = out_laplacian2

    # Write out the result
    output_name = sys.argv[5] + "output_cylindrical_lpb.png"
    cv2.imwrite(output_name, output_image)

    return True


'''
This exact function will be used to evaluate your results for HW2
Compare your result with master image and get the difference, the grading
criteria is posted on Piazza
'''


def RMSD(target, master):
    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or
            master_channel != target_channel):
        return -1
    else:
        total_diff = 0.0
        master_channels = cv2.split(master)
        target_channels = cv2.split(target)

        for i in range(0, len(master_channels), 1):
            dst = cv2.absdiff(master_channels[i], target_channels[i])
            dst = cv2.pow(dst, 2)
            mean = cv2.mean(dst)
            total_diff = total_diff + mean[0]**(1 / 2.0)

        return total_diff


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) != 6):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    input_image1 = cv2.imread(sys.argv[2], 0)
    input_image2 = cv2.imread(sys.argv[3], 0)
    input_image3 = cv2.imread(sys.argv[4], 0)
    
    function_launch = {
       1 : Perspective_warping,
       2 : Cylindrical_warping,
       3 : Bonus_perspective_warping,
       4 : Bonus_cylindrical_warping 
       }

       # Call the function
    function_launch[question_number](input_image1, input_image2, input_image3)

