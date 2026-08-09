import os
import shutil
import fastapi
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from gradio_client import Client, handle_file

app = FastAPI()

# Kết nối tới mô hình AI CodeFormer
try:
    client = Client("sczhou/CodeFormer")
    print("Successfully connected to sczhou/CodeFormer")
except Exception as e:
    print(f"Error connecting to sczhou/CodeFormer: {e}")
    client = None

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "vercel_timeout": "30s_config_active"}

@app.post("/api/enhance")
async def enhance_image(file: UploadFile = File(...)):
    if client is None:
        raise HTTPException(status_code=503, detail="Could not connect to CodeFormer model service.")
    
    temp_input_path = f"/tmp/{file.filename}"
    
    try:
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save temporary image file: {e}")
    
    try:
        # Gọi model CodeFormer
        result = client.predict(
            handle_file(temp_input_path), # Ảnh đầu vào
            0.7,                          # CodeFormer fidelity (0.0 -> 1.0)
            True,                         # Background enhance
            True,                         # Face upsample
            2                             # Upscale factor
        )
        
        # Bóc tách đường dẫn ảnh nếu Gradio trả về tuple hoặc list
        final_path = result[0] if isinstance(result, (tuple, list)) else result
        
        if isinstance(final_path, dict) and "path" in final_path:
            final_path = final_path["path"]

        # Kiểm tra file và trả kết quả về
        if final_path and os.path.exists(str(final_path)):
            return FileResponse(str(final_path), media_type="image/jpeg", filename=f"enhanced_{file.filename}")
        else:
            return JSONResponse(status_code=500, content={"detail": f"AI output valid path not found. Raw output: {str(result)}"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"AI enhancement failed: {str(e)}"})

    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)