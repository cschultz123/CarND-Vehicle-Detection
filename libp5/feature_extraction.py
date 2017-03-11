import cv2
import numpy as np
from skimage.feature import hog

from libp5 import image_utils


def cvt_color(image, cspace="RGB"):
    """
    Convert image to new color space.

    Note: This does not modify the original image.

    Supported color spaces are:

        1. HSV
        2. LUV
        3. HLS
        4. YUV
        5. YCrCb

    :param image: (ndarray) image
    :param cspace: (str) color space
    :return: (ndarray) converted image
    :raises: (Exception) unsupported color space
    """
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            raise Exception("Unsupported colorspace {}.".format(cspace))
    else:
        feature_image = np.copy(image)
    return feature_image


def bin_spatial(img, size=(32, 32)):
    """
    This function resizes the image and flattens it to create the feature
    vector.

    :param img: (ndarray) source image
    :param size: (2-tuple) target height and width
    :return: (1D ndarray) feature vector
    """
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    This method takes a three channel image as input, computes the histogram
    of each channel using the specified number of bins and bin range. The
    three histogram are concatenated and returned as the feature vector.

    :param img: (ndarray) 3-channel source image
    :param nbins: (int) number of bins for histogram
    :param bins_range: (2-tuple) minimum bin value, maximum bin value
    :return: (1D ndarray) feature vector
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def apply_hog(img, orient, pix_per_cell, cell_per_block,
              vis=False, feature_vec=True):
    """
    Computes histogram of oriented gradients feature vector from image.

    :param img: (2D ndarray) source image
    :param orient: (int) number of orientation bins
    :param pix_per_cell: (int) size of cell in which HOG is computed
    :param cell_per_block: (int) size of cell block used for nomalization
    :param vis: (bool) return image for visualization
    :param feature_vec: (bool) ravel HOG feature
    :return: (1-tuple or 2-tuple) feature vector, (optional) visualization image
    """
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def get_hog_features(img, orient, pix_per_cell, cell_per_block, hog_channel):
    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(img.shape[2]):
            hog_features.append(apply_hog(img[:, :, channel],
                                          orient, pix_per_cell, cell_per_block,
                                          vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = apply_hog(img[:, :, hog_channel], orient,
                                 pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    return hog_features


def extract_feature(image, cspace, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block,
                    hog_channel, spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Returns concatenated spatial, color histogram, and hog feature vector for
    image.

    This method implements the feature extraction pipeline. This implemented
    in the following manner.

        1. Convert image to specified color space.
        2. Compute spatial feature vector.
        3. Compute color histogram feature vector.
        4. Compute hog feature vector.
        5. Concatenate feature vectors.

    :param image: (ndarray) 3-channel image
    :param cspace: (str) color space
    :param spatial_size: (int) resized image dimensions
    :param hist_bins: (int) number of bins for color histograms
    :param hist_range: (2-tuple) min, max bin values for color histograms
    :param orient: (int) number of orientation bins for HOG
    :param pix_per_cell: (int) dimensions of cell for HOG
    :param cell_per_block: (int) dimensions of normalization block for HOG
    :param hog_channel: (int or str) color channel to use for HOG or 'ALL' to use all channels.
    :param spatial_feat: (bool) compute spatial feature
    :param hist_feat: (bool) compute histogram feature
    :param hog_feat: (bool) compute HOG feature
    :return: (1D ndarray) image feature vector
    """
    feature_image = image.astype(np.uint8)
    feature_image = cvt_color(feature_image, cspace)

    spatial_features = np.array([])
    if spatial_feat:
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)

    hist_features = np.array([])
    if hist_feat:
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

    hog_features = np.array([])
    if hog_feat:
        # Apply apply_hog() to get hog features
        hog_features = get_hog_features(feature_image, orient, pix_per_cell, cell_per_block, hog_channel)

    combined_features = np.concatenate((spatial_features, hist_features, hog_features))

    return combined_features


def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                         hist_bins=32, hist_range=(0, 256), orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel=0,
                         spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Returns list of feature vectors of images.

    This method applies extract_feature to each image in the list and returns
    a list of the feature vectors.

    :param imgs: (list) image filenames (must be valid!)
    :param cspace: (str) color space
    :param spatial_size: (int) resized image dimensions
    :param hist_bins: (int) number of bins for color histograms
    :param hist_range: (2-tuple) min, max bin values for color histograms
    :param orient: (int) number of orientation bins for HOG
    :param pix_per_cell: (int) dimensions of cell for HOG
    :param cell_per_block: (int) dimensions of normalization block for HOG
    :param hog_channel: (int or str) color channel to use for HOG or 'ALL' to use all channels.
    :return: (list) image feature vectors
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = image_utils.imread(file)

        # compute and combine features into single feature vector
        combined_features = extract_feature(image, cspace, spatial_size,
                                            hist_bins, hist_range, orient,
                                            pix_per_cell, cell_per_block,
                                            hog_channel, spatial_feat,
                                            hist_feat, hog_feat)

        if combined_features.size == 0:
            raise Exception("Empty feature vector. Make sure that at least "
                            "one of the features is enabled.")

        # Append the new feature vector to the features list
        features.append(combined_features)

    # Return list of feature vectors
    return features


