import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def main(path):

    Path(f"{path}/cropped").mkdir(parents=True, exist_ok=True)

    for file in tqdm(glob.glob(f"{path}/*")):

        if file.endswith(".png"):

            file_name = file.split("/")[-1].split(".")[0]

            im = Image.open(file)
            width, height = im.size
            im = im.crop((99, 190, width - 83, 297))
            im = np.array(im)
            im = Image.fromarray(im)
            width, height = im.size

            print(f"{path}/cropped/{file_name}_1.png")

            # 1ST IMAGE
            im1 = im.crop((0, 0, 108, height))
            im1 = np.array(im1)
            im1 = Image.fromarray(im1)
            im1.save(f"{path}/cropped/{file_name}_1.png")

            # 2ND IMAGE
            im2 = im.crop((175, 0, 283, height))
            im2 = np.array(im2)
            im2 = Image.fromarray(im2)
            im2.save(f"{path}/cropped/{file_name}_2.png")

            # 3RD IMAGE
            im3 = im.crop((350, 0, width, height))
            im3 = np.array(im3)
            im3 = Image.fromarray(im3)
            im3.save(f"{path}/cropped/{file_name}_3.png")


if __name__ == "__main__":

    path = list(sys.argv)[-1]
    main(path)
