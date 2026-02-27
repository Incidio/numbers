
from fastapi import FastAPI, UploadFile
from predict import process

app = FastAPI()

@app.get('/')
def root():
    return {'message': 'Hello, World!'}

@app.post('/predict')
async def predict(file: UploadFile):
    image_bytes = await file.read()
    prediction = process(image_bytes)
    return {'prediction': prediction}
