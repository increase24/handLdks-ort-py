import cv2
import numpy as np
import queue
import time

from threading import Thread

class VideoStream(Thread):
    """
    Thread that reads frames from the video source at a given frame rate.
    """
    def __init__(self, video_source, fps, queue_size=1):
        Thread.__init__(self)
        self.video_source = cv2.VideoCapture(video_source)
        self.frames = queue.Queue(queue_size)
        self.fps = fps
        self.delta_t = 1.0 / self.fps
        self._shutdown = False
    
    def stop(self):
        """Stop the VideoStream instance."""
        self._shutdown = True

    def get_image(self):
        if self.frames.empty():
            ret = False
            frame = None
        else:
            ret = True
            frame = self.frames.get()
        return ret, frame

    def run(self):
        while not self._shutdown:
            start_time = time.perf_counter()
            ret, image_tuple = self.video_source.read()
            if self.frames.full():
                self.frames.get()
            # last frame is None
            if image_tuple is None:
                self.stop()
                continue
            self.frames.put(image_tuple, False)
            # wait before the next framegrab to enforce a certain FPS
            elapsed = time.perf_counter() - start_time
            delay = self.delta_t - elapsed
            if delay > 0:
                time.sleep(delay) 
            






