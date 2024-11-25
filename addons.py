import time
from langchain.agents import tool
import cv2
import numpy as np
from cv2 import VideoCapture
from threading import Thread, Lock
import base64
from typing import Union

class WebcamStream:
    def __init__(self, src: int = 0):
        self.stream = VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        
        self.frame = None
        self.stopped = False
        self.lock = Lock()
        
        Thread(target=self._update, daemon=True).start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                continue
                
            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    async def read(self, encode: bool = False) -> Union[np.ndarray, bytes, None]:
        self.lock.acquire()
        frame = self.frame.copy() if self.frame is not None else None
        self.lock.release()

        if frame is None:
            return None

        if encode:
            params = [
                cv2.IMWRITE_JPEG_QUALITY, 95,  # Higher JPEG quality
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,  # Enable optimization
                cv2.IMWRITE_JPEG_PROGRESSIVE, 1  # Use progressive JPEG
            ]
            _, buffer = cv2.imencode('.jpg', frame, params)
            return base64.b64encode(buffer.tobytes()) if buffer is not None else None

        return frame

    async def stop(self):
        self.stopped = True
        self.stream.release()

@tool
def get_current_time() -> str:
    """Get the current time. No arguments are required."""
    return time.strftime('%I:%M %p')
