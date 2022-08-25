# De Bruits - Computer Vision and motion

Run the following commands on Google Colab:

```ruby
!git clone https://github.com/fabiocuri/debruits.git

from google.colab import drive
drive.mount('/content/drive')

!bash /content/debruits/run.sh create

!bash /content/debruits/run.sh preprocess

!bash /content/debruits/run.sh train start

!bash /content/debruits/run.sh train continue

!bash /content/debruits/run.sh inference

!bash /content/debruits/run.sh crop /content/drive/MyDrive/_plots

!bash /content/debruits/run.sh super_resolution /content/drive/MyDrive/output/inference
```
