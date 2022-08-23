import sys
import glob
from tqdm import tqdm

import numpy as np
from PIL import Image

from rdn import RDN

if __name__ == "__main__":

    n_loop = int(sys.argv[-1])

    rdn = RDN(weights="psnr-small")

    for file in tqdm(glob.glob(f"../data/output/inference/*")):

        for _ in range(n_loop):

            img = Image.open(file)
            lr_img = np.array(img)

            sr_img = rdn.predict(lr_img)
            highres_img = Image.fromarray(sr_img)
            highres_img.save(file)
