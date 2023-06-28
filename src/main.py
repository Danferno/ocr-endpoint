# uvicorn src.main:app --reload
from fastapi import FastAPI, Response, UploadFile
from pydantic import BaseModel
import easyocr
import numpy as np
import cv2 as cv
import json
import io
import pickle
from datetime import datetime

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
    text = reader.readtext(image=img_array, batch_size=60, detail=1)

    with io.BytesIO() as buffer:
        pickle.dump(text, buffer)
        text_bytes = buffer.getvalue()

    return Response(text_bytes)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True)