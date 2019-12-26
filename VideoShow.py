from threading import Thread
import cv2 as cv
from imutils.video import FPS


class VideoShow:
    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False
        self.fps = FPS().start()

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            self.fps.update()
            cv.imshow("Video", self.frame)
            if cv.waitKey(1) == ord('q'):
                self.stopped = True

    def stop(self):
        self.fps.stop()
        self.stopped = True
        cv.destroyAllWindows()
