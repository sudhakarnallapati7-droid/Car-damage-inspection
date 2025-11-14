from fastapi import FastAPI, File, UploadFile
from fastapi_server.model_helper import predict
import os

app = FastAPI()

@app.get("/")
def root():
    return {"message": "FastAPI Car Damage Model is running ✅"}

@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_path = "uploaded_image.jpg"

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        prediction = predict(image_path)
        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}
