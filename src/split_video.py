from handlers import VideoClass
import sys

if __name__ == "__main__":

  video_path = sys.argv[-1]
  video = VideoClass(video_path)
  video.read_video()
  video.get_video_name()
  video.video2frames("/".join(sys.argv[-1].split("/")[:-1]), 3)
