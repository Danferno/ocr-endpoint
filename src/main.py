# uvicorn src.main:app --reload
from fastapi import FastAPI, Response, UploadFile
from pydantic import BaseModel
import easyocr
import cv2 as cv
import numpy as np
import io
import pickle
from datetime import datetime

from torch.cuda import OutOfMemoryError as TorchOutOfMemoryError

class OcrRequest(BaseModel):
    img_array: list
    batch_size: int | None = 60
    detail: int | None = 1

app = FastAPI()
reader = easyocr.Reader(lang_list=['nl', 'fr', 'de', 'en'])

@app.get('/')
async def root():
    return {'message': 'Hello World'}

@app.post('/easyocr')
async def ocr_easyocr(img_array_pkl: UploadFile):
    print(datetime.now(), 'Request arrived')
    img_array = np.array(pickle.loads(await img_array_pkl.read()), dtype=np.uint8)
    try:
        text = reader.readtext(image=img_array, batch_size=60, detail=1)
    except TorchOutOfMemoryError:   # type:ignore
        rescale_factor = 3
        img_array_dX = cv.resize(img_array, (0,0), fx=1/rescale_factor, fy=1/rescale_factor, interpolation=cv.INTER_LANCZOS4)
        try:
            text_dX = reader.readtext(image=img_array_dX, batch_size=60, detail=1)
        except TorchOutOfMemoryError:   # type:ignore
            rescale_factor = 5
            img_array_dX = cv.resize(img_array, (0,0), fx=1/rescale_factor, fy=1/rescale_factor, interpolation=cv.INTER_LANCZOS4)

        text = []
        for dX in text_dX:      # d4 = text_d4[0]
            bbox = dX[0]
            bbox = [[x*rescale_factor for x in L] for L in bbox]
            d1 = (bbox, *dX[1:])
            text.append(d1)

    with io.BytesIO() as buffer:
        pickle.dump(text, buffer)
        text_bytes = buffer.getvalue()

    return Response(text_bytes)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True)