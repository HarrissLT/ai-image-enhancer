import os
import shutil
import fastapi
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from gradio_client import Client, handle_file

# Khởi tạo FastAPI
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
    """Endpoint test nhanh xem Serverless Function có sống không."""
    return {"status": "ok", "vercel_timeout": "30s_config_active"}

@app.post("/api/enhance")
async def enhance_image(file: UploadFile = File(...)):
    # 1. Kiểm tra kết nối model
    if client is None:
        raise HTTPException(status_code=503, detail="Could not connect to CodeFormer model service.")
    
    # 2. Tạo đường dẫn file tạm
    temp_input_path = f"/tmp/{file.filename}"
    
    # 3. Lưu file tạm gửi từ iPhone
    try:
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save temporary image file: {e}")
    
    # 4. Xử lý qua AI CodeFormer (Khử mờ, làm nét mặt & background)
    # Tăng thời gian xử lý: config `maxDuration` trong vercel.json đã được đặt
    try:
        # Gọi model với gradio_client
        # process_image_filepath = client.predict(...)
        
        # Sửa thành call cụ thể hơn để test timeout
        result_path = client.predict(
            image=handle_file(temp_input_path),
            codeformer_fidelity=0.7,   # Cân bằng giữa độ nét và giữ nét mặt gốc
            background_enhance=True,   # Làm nét cả nền ảnh
            face_upsample=True,        # Tăng độ phân giải khuôn mặt
            upscale=2,                 # Tăng gấp đôi độ phân giải
            api_name="/predict"
        )
        
        # Trả file ảnh đã làm nét về iPhone trực tiếp
        if os.path.exists(result_path):
            return FileResponse(result_path, media_type="image/jpeg", filename=f"enhanced_{file.filename}")
        else:
            return JSONResponse(status_code=500, content={"detail": "AI processing successful but result file not found."})

    except Exception as e:
        # Bắt lỗi timeout hoặc lỗi từ Hugging Face
        error_message = str(e)
        if "Function invocation has timed out" in error_message or "time limit" in error_message:
            return JSONResponse(status_code=504, content={"detail": f"Model processing timed out (>30s): {e}. Try a smaller image."})
        else:
            return JSONResponse(status_code=500, content={"detail": f"AI enhancement failed: {e}. If this is a time-out, ensure you pushed vercel.json with maxDuration: 30."})

    finally:
        # 5. Luôn dọn dẹp file tạm đầu vào
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)