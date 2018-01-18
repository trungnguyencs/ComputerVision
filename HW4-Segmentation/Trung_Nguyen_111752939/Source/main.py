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

if __name__ == '__main__':

    # validate the input arguments
    # if (len(sys.argv) != 4):
    #     help_message()
    #     sys.exit()
    # img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    # img_marking = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    img = cv2.imread("astronaut.png", cv2.IMREAD_COLOR)
    img_marking = cv2.imread("astronaut_marking.png", cv2.IMREAD_COLOR)
    # img_marking = 255 - cv2.dilate(255 - img_marking, None)
    # img_marking = 255 - cv2.dilate(255 - img_marking, None)

    img = cv2.GaussianBlur(img, (9,9), 0.6)

    centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(img)
    norm_hists = normalize_histograms(color_hists)

    fg_segments, bg_segments = find_superpixels_under_marking(img_marking, superpixels)
    fgbg_superpixels = [fg_segments, bg_segments]
    fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, color_hists)     # cum hist of FG
    bg_cumulative_hist = cumulative_histogram_for_superpixels(bg_segments, color_hists)  # cum hist of BG
    fgbg_hists = [fg_cumulative_hist, bg_cumulative_hist]
    graph_cut = do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors)

    segmask = pixels_for_segment_selection(superpixels, np.nonzero(graph_cut))
    segmask = np.uint8(segmask * 255)

    # fullMask = np.zeros(img.shape)
    # fullMask[:,:,0] = fullMask[:,:,1] = fullMask[:,:,2] = segmask/255

    # read video file. Save result
    # output_name = sys.argv[3] + "mask.png"

    output_name = "mask.png"
    cv2.imwrite(output_name, segmask)

    out_sample = cv2.imread("example_output.png", cv2.IMREAD_GRAYSCALE)
    print "out_sample chanel: ", np.shape(out_sample)
    print "RMSD: ", RMSD(segmask, out_sample)

    # plt.subplot(131); plt.imshow(img)
    # plt.subplot(132); plt.imshow(segmask)
    # plt.subplot(133); plt.imshow(out_sample)
    # ## plt.subplot(144); plt.imshow(np.uint8(fullMask*img))
    # plt.show()