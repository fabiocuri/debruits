import glob
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

from handlers import ImageClass
from rdn import RDN

if __name__ == "__main__":

    n_loop = int(sys.argv[-1])
    path = sys.argv[-2]

    rdn = RDN(weights="noise-cancel")

    for file in tqdm(glob.glob(f"{path}/*")):

        for _ in range(n_loop):

            img = Image.open(file)
            lr_img = np.array(img)

            if lr_img.shape[2] == 4:

                lr_img = lr_img[:,:,:3]

            sr_img = rdn.predict(lr_img)
            highres_img = Image.fromarray(sr_img)
            highres_img.save(file)
            
