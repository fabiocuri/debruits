from pathlib import Path
import cv2
import os


class VideoClass:
    """ 
    A class that reads, preprocesses and converts videos.
    """

    def __init__(self, input_path='./'):

        super(VideoClass, self).__init__()
        self.input_path = input_path

    def read_video(self):

        self.video = cv2.VideoCapture(self.input_path)
        
    def get_video_length(self):
    
        fps = self.video.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps
        
        return duration%60

    def get_video_name(self):

        base = os.path.basename(self.input_path)
        self.video_name = os.path.splitext(base)[0]

    def resize_image(self, image, dim):

        return cv2.resize(image, dim)

    def video2frames(self, output_path, dim):
    
        output_path = f"{output_path}/{self.video_name}"
        Path(output_path).mkdir(parents=True, exist_ok=True)

        success, image = self.video.read()
        image = self.resize_image(image, dim)
        count = 1

        while success:
        
            cv2.imwrite(f"{output_path}/frame_{count}.jpg", image)
                
            success, image = self.video.read()
            
            try:
            
                image = self.resize_image(image, dim)
                count += 1
                
            except:
            
                pass
                
    def imshow(self):
    
        while(self.video.isOpened()):
        
            ret, frame = self.video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame',gray)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
            
                break

        self.video.release()
        cv2.destroyAllWindows()


class ImageClass:
    """ 
    A class that reads, preprocesses and converts images.
    """

    def __init__(self, input_path='./', cv2image=None):

        super(ImageClass, self).__init__()
        self.input_path = input_path
        self.cv2image = cv2image

    def read_image(self, flag=cv2.IMREAD_COLOR):

        if self.cv2image is not None:

            self.image = self.cv2image

        else:

            self.image = cv2.imread(self.input_path, flag)

    def get_image_name(self, image_name=None):

        if image_name is not None:

            self.image_name = image_name
            self.parent_folder = ''

        else:

            self.image_name = os.path.basename(self.input_path)
            path = Path(self.input_path)
            self.parent_folder = os.path.basename(path.parent)

    def edges_canny(self):

        self.image = cv2.Canny(self.image, 100, 200)

    def export_image(self, output_path):

        output_path = f"{output_path}/{self.parent_folder}"
        Path(output_path).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(f"{output_path}/{self.image_name}", self.image)

    def imshow(self):

        cv2.imshow('image', self.image)
        cv2.waitKey(0)
