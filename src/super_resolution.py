import glob
import sys

import cv2
from tqdm import tqdm

if __name__ == "__main__":

    n_loop = int(sys.argv[-1])
    path = sys.argv[-2]

    for file in tqdm(glob.glob(f"{path}/*")):

        for _ in range(n_loop):

            img = cv2.imread(file)
            dst = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 5, 10)
            cv2.imwrite(file, dst)
