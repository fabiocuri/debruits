import glob
import sys

import cv2
import numpy as np
from ISR.models import RDN
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":

    n_loop = int(sys.argv[-1])
    path = sys.argv[-2]

    rdn = RDN(weights="psnr-small")

    for file in tqdm(glob.glob(f"{path}/*")):

        for _ in range(2):

            img = Image.open(file)
            lr_img = np.array(img)

            sr_img = rdn.predict(lr_img)
            img = Image.fromarray(sr_img)
            img = img.save(file)

        for _ in range(n_loop):

            img = cv2.imread(file)
            dst = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 5, 10)
            cv2.imwrite(file, dst)
