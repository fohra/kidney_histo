# Code from Joona Pohjonen @jopo666
import random
from PIL import ImageFilter

class GaussianBlur(object):
    """Gaussian Blur for a PIL image.

    Args:
        p (float, optional): 
            Probability of applying.
        radius_min (float, optional): 
            For ImageFilter.GaussianBlur. Defaults to 0.1.
        radius_max (float, optional):
            For ImageFilter.GaussianBlur. Defaults to 2.0.
    """

    def __init__(self, p: float, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.min = radius_min
        self.max = radius_max

    def __call__(self, img):
        if random.random() <= self.p:
            img.filter(ImageFilter.GaussianBlur(
                radius=random.uniform(self.min, self.max)
            ))
            return img
        else:
            return img
