import cv2
import threading
import numpy as np
import imutils
from gst_cam import CSI_Camera
from panorama import Stitcher

# TODO: Refactor code

left_camera = None
right_camera = None

stitcher = Stitcher()
total = 0

def gstreamer_pipeline(
    sensor_id=0,
    sensor_mode=3,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            sensor_mode,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )



def main():
    left_camera = CSI_Camera()
    left_camera.open(
        gstreamer_pipeline(
            sensor_id=1,
            sensor_mode=3,
            flip_method=0,
            display_height=1080,
            display_width=1920,
        )
    )
    left_camera.start()

    right_camera = CSI_Camera()
    right_camera.open(
        gstreamer_pipeline(
            sensor_id=0,
            sensor_mode=3,
            flip_method=0,
            display_height=1080,
            display_width=1920,
        )
    )
    right_camera.start()
    while True:
        _, left = left_camera.read()
        _, right = right_camera.read()
        left = imutils.resize(left, width=640, height=480)
        right = imutils.resize(right, width=640, height=480)
        result = stitcher.stitch([left, right])
        cv2.imshow("Result", result)
        cv2.imshow("Left Frame", left)
        cv2.imshow("Right Frame", right)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    print("[INFO] cleaning up...")
    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()
    cv2.destroyAllWindows()


def only_images():
    imageA = cv2.imread("2.jpg") # image 1
    imageB = cv2.imread("1.jpg") # image 2
    imageA = imutils.resize(imageA, width=400)
    imageB = imutils.resize(imageB, width=400)

    (result, vis) = stitcher.stitch_image([imageA, imageB], showMatches=True)
    cv2.imshow("ImageA", imageA)
    cv2.imshow("ImageB", imageB)
    cv2.imshow("Keypoints Matches", vis)
    cv2.imshow("Result", result)
    cv2.waitKey(0)



if __name__ == "__main__":
    main()
    # only_images()
