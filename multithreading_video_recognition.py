from VideoGet import VideoGet
from VideoShow import VideoShow
import cv2 as cv
from imutils.video import FPS
import argparse

fps = FPS().start()

video_getter = VideoGet(0).start()

while True:
    if cv.waitKey(1) == ord('q') or video_getter.stopped:
        video_getter.stop()
        break

    frame = video_getter.frame
    fps.update()
    cv.imshow("Video", frame)

fps.stop()
print(fps.elapsed())
print(fps.fps())

# cap = cv.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#
#     if cv.waitKey(1) == ord('q'):
#         break
#     fps.update()
#     cv.imshow("Video", frame)
#
#
# fps.stop()
# print(fps.elapsed())
# print(fps.fps())
#
# cv.destroyAllWindows()

# video_getter = VideoGet(0).start()
# video_shower = VideoShow(video_getter.frame).start()
#
# while True:
#     if video_getter.stopped or video_shower.stopped:
#         video_getter.stop()
#         video_shower.stop()
#         break
#
#     frame = video_getter.frame
#     video_shower.frame = frame
#

# print(video_shower.fps.elapsed())
# print(video_shower.fps.fps())
# # no_threading_get(0)
# threading_get(0)
