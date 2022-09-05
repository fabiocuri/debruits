# De Bruits - Computer Vision and motion

## Image

Run the following commands on Google Colab for the entire pipeline:

```
!git clone https://github.com/fabiocuri/debruits.git

from google.colab import drive
drive.mount('/content/drive')

!bash /content/debruits/run_image.sh create

!bash /content/debruits/run_image.sh preprocess

!bash /content/debruits/run_image.sh train start

!bash /content/debruits/run_image.sh train continue

!bash /content/debruits/run_image.sh inference

!bash /content/debruits/run_image.sh super_resolution /content/drive/MyDrive/image/output/inference

!bash /content/debruits/run_image.sh crop /content/drive/MyDrive/image/plots

!bash /content/debruits/run_image.sh super_resolution /content/drive/MyDrive/image/plots/cropped

```

In order to split videos into frames:

```
!bash /content/debruits/run_image.sh split_video /content/drive/MyDrive/video/DSC_0022.m4v
```

## Audio

Run the following commands on Google Colab: