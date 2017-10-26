# ================================================
# Skeleton codes for HW4
# Read the skeleton codes carefully and put all your
# codes into main function
# ================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay
import os
import sys

def help_message():
    print("Usage: [Input_Image] [Input_Marking] [Output_Directory]")
    print("[Input_Image]")
    print("Path to the input image")
    print("[Input_Marking]")
    print("Path to the input marking")
    print("[Output_Directory]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " astronaut.png " + "astronaut_marking.png " + "./")


# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=500, compactness=20)        # size of the input images

    # print segments
    # print "size of img: ", np.shape(img)
    # print "size of segments: ", np.shape(segments)

    segments_ids = np.unique(segments)      # return unique element of segments
    # print segments_ids
    # print "number of segments: ", len(segments_ids)

    # centers of each segment. coordinates.
    centers = np.array(
        [np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])


    # plt.imshow(segments, cmap='gray')
    # plt.scatter(centers[:,0], centers[:,1])
    # plt.show()
    # print centers
    # print "size of centers: ", np.shape(centers)

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [30, 30]  # H = S = 20
    ranges = [0, 360, 0, 1]  # H: [0, 360], S: [0, 1]

    # colors_hists: #rows = #row of segment. 452x400
    colors_hists = np.float32([
        cv2.calcHist([hsv], [0, 1], np.uint8(segments == i), bins,
                     ranges).flatten() for i in segments_ids
    ])

    # print "size of colors_hist: ", np.shape(colors_hists)
    # print "colors_hists: ", colors_hists

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    # print "tri: ", tri.vertex_neighbor_vertices
    # print "size of tri: ", np.shape(tri.vertex_neighbor_vertices)
    return (centers, colors_hists, segments, tri.vertex_neighbor_vertices)

# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:, :, 0] != 255])       # foreground
    bg_segments = np.unique(superpixels[marking[:, :, 2] != 255])       # background
    return (fg_segments, bg_segments)

# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids], axis=0)
    return h / h.sum()

# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)     # Find the indices of elements
    return pixels_mask

# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])

# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr, indices = neighbors
    for i in range(len(indptr) - 1):
        N = indices[indptr[i]:indptr[i + 1]]  # list of neighbor superpixels
        hi = norm_hists[i]  # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]  # histogram for neighbor
            g.add_edge(nodes[i], nodes[n],
                       20 - cv2.compareHist(hi, hn, hist_comp_alg),
                       20 - cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i, h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 2000)  # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 2000, 0)  # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i],
                        cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                        cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)

def RMSD(target, master):
    # Note: use grayscale images only

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
        dst = cv2.absdiff(master, target)
        dst = cv2.pow(dst, 2)
        mean = cv2.mean(dst)
        total_diff = mean[0]**(1 / 2.0)

        return total_diff



def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode, fg_segs, bg_segs
    r = 5

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                w, h = np.shape(img)[0], np.shape(img)[1]
                for i in range(-r, r+1):
                    for ii in range(-r, r + 1):
                        y_i = min(np.max(y+i, 0), min(y+i, w-1))
                        x_ii = min(max(x+ii, 0), min(x+ii, h-1))
                        fg_segs[y_i, x_ii] = 0
                        bg_segs[y_i, x_ii] = 255
            else:
                cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
                w, h = np.shape(img)[0], np.shape(img)[1]
                for i in range(-r, r + 1):
                    for ii in range(-r, r + 1):
                        y_i = min(max(y+i, 0), min(y+i, w-1))
                        x_ii = min(max(x+ii, 0), min(x+ii, h-1))
                        bg_segs[y_i, x_ii] = 0
                        fg_segs[y_i, x_ii] = 255

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


if __name__ == '__main__':

    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    print "\n\n\nHow to annotate image: \n"
    print "By default, annotate foreground first (Red color)"
    print "At any time, press F on keyboard to trigger annotation for foreground"
    print "and press B on keyboard to trigger annotation for background"
    print "Press escape to exit!\n"
    print "Segmentation only starts when there are both FG and BG annotation"

    drawing = False  # true if mouse is pressed
    mode = True  # if True, draw Foreground, press b/B to switch to background
    ix, iy = -1, -1
    # mouse callback function

    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

    fg_segs = np.uint8(np.ones((np.shape(img)[0], np.shape(img)[1])) * 255)
    bg_segs = np.uint8(np.ones((np.shape(img)[0], np.shape(img)[1])) * 255)
    img_blur = cv2.GaussianBlur(img, (9, 9), 0.6)

    # precompute the superpixel for the image 1 time.
    centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(img_blur)
    norm_hists = normalize_histograms(color_hists)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    img_marking = np.zeros(img.shape)

    import time
    start = time.time()

    while (1):
        cv2.imshow('image', img)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('b') or k == ord('B'):
            mode = False
        if k == ord('f') or k == ord('F'):
            mode = True
        elif k == 27:
            break

        if time.time() - start > 0.25:       # delay 0.25 seconds to process new data
            start = time.time()
            img_marking[:, :, 2] = bg_segs; img_marking[:, :, 0] = fg_segs

            if np.sum(fg_segs - 255) > 10 and np.sum(bg_segs - 255) > 10:
                fg_segments, bg_segments = find_superpixels_under_marking(img_marking, superpixels)

                fgbg_superpixels = [fg_segments, bg_segments]

                fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, color_hists)     # cum hist of FG
                bg_cumulative_hist = cumulative_histogram_for_superpixels(bg_segments, color_hists)  # cum hist of BG
                fgbg_hists = [fg_cumulative_hist, bg_cumulative_hist]
                graph_cut = do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors)

                segmask = pixels_for_segment_selection(superpixels, np.nonzero(graph_cut))
                segmask = np.uint8(segmask * 255)

                output_name = "mask_bonus.png"
                cv2.imshow("mask", segmask)
                cv2.imwrite(output_name, segmask)

                # out_sample = cv2.imread("example_output.png", cv2.IMREAD_GRAYSCALE)
                # print "out_sample chanel: ", np.shape(out_sample)
                # print "RMSD: ", RMSD(segmask, out_sample)


    cv2.destroyAllWindows()