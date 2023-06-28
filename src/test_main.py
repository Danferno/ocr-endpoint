from PIL import Image
import cv2 as cv
import numpy as np
import pickle
import io
import pytest
import json
import requests

# Test
if __name__ == '__main__':
    with io.BytesIO() as buffer:
        pickle.dump(cv.imread(r"C:\Users\Jessa\Pictures\newTable7.png", flags=cv.IMREAD_GRAYSCALE), buffer)
        img_array_bytes = buffer.getvalue()

    response = requests.post(url="http://127.0.0.1:8000/easyocr", files={'img_array_pkl': img_array_bytes})
    text = pickle.loads(response.content)
    assert isinstance(text, list)
    assert len(text) == 57