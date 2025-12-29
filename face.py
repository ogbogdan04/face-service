from fastapi import FastAPI, UploadFile, File
import face_recognition
import numpy as np
import io

app = FastAPI()

def get_encoding(image_bytes: bytes):
    image = face_recognition.load_image_file(io.BytesIO(image_bytes))
    encodings = face_recognition.face_encodings(image)

    if len(encodings) != 1:
        return None

    return encodings[0]

@app.post("/verify")
async def verify(
    indexImage: UploadFile = File(...),
    selfieImage: UploadFile = File(...)
):
   
    index_bytes = await indexImage.read()
    selfie_bytes = await selfieImage.read()

    index_encoding = get_encoding(index_bytes)
    selfie_encoding = get_encoding(selfie_bytes)

    if index_encoding is None or selfie_encoding is None:
        return {
            "error": "Na jednoj od slika nije detektovano tačno jedno lice."
        }

    distance = np.linalg.norm(index_encoding - selfie_encoding)
    is_match = distance <= 0.6

    return {
        "distance": float(distance),
        "isMatch": bool(is_match)
    }
