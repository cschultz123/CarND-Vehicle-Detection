import glob
from sklearn.utils import shuffle


def get_vehicle_classification_data():
    """
    Return list of file paths corresponding to vehicle and non-vehicle images.

    All of the images in this dataset have shape, 64 x 64 x 3.

    :return: (2-tuple) list of vehicle images, list of non-vehicle images
    """
    vehicles = glob.glob("vehicles/*/*.png")
    shuffle(vehicles)

    non_vehicles = glob.glob("non-vehicles/*/*.png")
    shuffle(non_vehicles)

    return vehicles, non_vehicles