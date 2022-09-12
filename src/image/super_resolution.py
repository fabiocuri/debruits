import glob
import sys

import cv2
import numpy as np
from PIL import Image
from rdn import RDN
from tqdm import tqdm

if __name__ == "__main__":

    n_loop = int(sys.argv[-1])
    path = sys.argv[-2]

    rdn = RDN(weights="psnr-small")

    for file in tqdm(glob.glob(f"{path}/*")):

        for _ in range(2):

            img = Image.open(file)
            img = np.array(img)
            img = img[:, :, :3]

            img = rdn.predict(img)
            img = img[:, :, :3]
            img = Image.fromarray(img)
            img = img.save(file)

        for _ in range(n_loop):

            img = cv2.imread(file)
            dst = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 5, 10)
            cv2.imwrite(file, dst)
