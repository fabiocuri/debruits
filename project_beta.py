from src.handlers.videohandlers import VideoClass

if __name__ == "__main__":

    # Semantic segmentation

    ## Read video and convert it into frames
    videohandler = VideoClass(video_path='./data/input/raw_videos/file_example_MP4_480_1_5MG.mp4', frames_path='./data/input/frames')
    videohandler.video2frames()
