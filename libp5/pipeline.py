import libp5.image_utils as utils
import numpy as np
import time


class VehicleDetector(object):

    def __init__(self, classifier, scaler, window_params, feature_params, consider_hist=True):
        # classification parameters
        self.clf = classifier
        self.scaler = scaler
        self.window_params = window_params
        self.feature_params = feature_params

        # cache windows from previous frame
        self.consider_hist = consider_hist
        self.window_history = []
        self.frames = 5

    def __call__(self, image):
        """
        This implements the vehicle detection pipeline.

        1. Generate windows to apply classifier in.
        2. Apply classifier to each window and store positive classifications.
        3. Generate heatmap from positive classification windows
        4. Threshold heatmap to remove noise and false positives.
        5. Draw bounding boxes around car regions.

        :param image: (ndarray) RGB image
        :param classifier: sklearn model
        :param scaler: sklearn scalerConso
        :return: (ndarray) image with bounding boxes around cars
        """
        imcpy = image.copy()
        # compute bounding boxes that classifier will be applied to
        windows = utils.slide_window(imcpy, **self.window_params)
        # apply classifier to image and return positive classifications
        on_windows = utils.search_windows(img=imcpy, windows=windows, clf=self.clf, scaler=self.scaler, **self.feature_params)
        # use previous frames to generate heatmap
        if self.consider_hist:
            # cache windows
            self.window_history.append(on_windows)
            # generate heatmap using windows from last n frames
            heatmap = np.zeros_like(image[:,:,0])
            for bboxes in self.window_history[-self.frames:]:
                # compute heatmap from positive classification windows
                heatmap += utils.compute_heatmap(imcpy, bboxes=bboxes)
        else:
            # only use current frame to generate heat map
            heatmap = utils.compute_heatmap(imcpy, bboxes=on_windows)
        # threshold heatmap
        heatmap = utils.threshold_heatmap(heatmap, threshold=5)
        # identify individual cars from heatmap
        return utils.draw_bounding_boxes(heatmap, imcpy, min_area=3000)








