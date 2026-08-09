import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from gradio_client import Client, handle_file

app = FastAPI()
client = Client("sczhou/CodeFormer")

@app.post("/api/enhance")
async def enhance(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    result = client.predict(
        image=handle_file(temp_path),
        codeformer_fidelity=0.7,
        background_enhance=True,
        face_upsample=True,
        upscale=2,
        api_name="/predict"
    )
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return FileResponse(result, media_type="image/jpeg")