import cv2
import numpy as np
from skimage.measure import label, regionprops
import time

from libp5 import feature_extraction


def imread(filename):
    """Read RGB image."""
    bgr = cv2.imread(filename)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_windows=((64, 64)), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Initialize a list to append window positions to
    window_list = []
    for xy_window in xy_windows:
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
        ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
        nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
        ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))

    # Return the list of windows
    return window_list


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, cspace='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = feature_extraction.extract_feature(
            test_img, cspace=cspace, spatial_size=spatial_size,
            hist_bins=hist_bins, hist_range=hist_range, orient=orient,
            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
            hog_channel=hog_channel, spatial_feat=spatial_feat,
            hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def compute_heatmap(img, bboxes):
    """
    Create mask where each element covered by a window is incremented by one.

    :param img: (ndarray) original image
    :param bboxes: (tuple) of window vertices
    :return: (ndarray) heatmap image
    """
    if len(img.shape) == 3:
        heat_mask = np.zeros_like(img[:,:,0])
    else:
        heat_mask = np.zeros_like(img)

    for bbox in bboxes:
        heat_mask[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1
    return heat_mask


def threshold_heatmap(heatmap, threshold):
    """
    Set all values in heatmap below specified threshold to zero.

    :param heatmap: (ndarray) heatmap of image
    :param threshold: (int) minimum element value to keep
    :return: (ndarray) thresholded heatmap
    """
    hm = heatmap.copy()
    # set all values below threshold to zero
    hm[heatmap <= threshold] = 0
    # set all values above threshold to one
    hm[hm > 0] = 1
    return hm


def draw_bounding_boxes(heatmap, image, min_area=4000):
    """
    Determine bounding boxes from filtered heat map.

    Arguments:
        heatmap (ndarray): heat map
        image (ndarray): original image
        min_area (int): minimum area of blob to keep

    Returns:
        image with bounding boxes drawn around cars
    """
    # assign number to blobs label image regions
    label_image, num_cars = label(input=heatmap, return_num=True)

    bboxes = []
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= min_area:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            bboxes.append(((minc, minr), (maxc, maxr)))

    return draw_boxes(img=image, bboxes=bboxes)