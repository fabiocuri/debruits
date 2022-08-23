import sys

import numpy as np
from PIL import Image

from rdn import RDN

if __name__ == "__main__":

    image_file = sys.argv[1]

    # improve resolution 5 times

    for _ in range(5):

        img = Image.open(f"../../../data/output/{image_file}")
        lr_img = np.array(img)

        rdn = RDN(weights="psnr-small")
        sr_img = rdn.predict(lr_img)
        highres_img = Image.fromarray(sr_img)
        highres_img.save(f"../../../data/output/{image_file}")
