import os
import json
import tempfile
import traceback
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException,BackgroundTasks
from starlette.background import BackgroundTask
from gradio_client import Client as GradioClient, handle_file
from fastapi.responses import FileResponse
import uuid
import threading
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from gradio_client.client import Job

jobs: Dict[str, Dict[str, Any]] = {}
GRADIO_APP_URL = "http://192.168.1.5:7862/"
gradio_client = None
app = FastAPI(
title="FramePack-eichi API",
description="一个通过FastAPI调用FramePack-eichi视频生成功能的接口。",
version="1.0.0"
)
def save_upload_file_to_temp(upload_file: UploadFile) -> str:
    """将上传的文件保存到临时文件并返回路径"""
    try:
        # 使用带后缀的临时文件，方便调试
        suffix = os.path.splitext(upload_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload_file.file.read())
        return tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理失败: {e}")

def watch_job_status(job: Job, job_id: str):
    """在一个后台线程中等待Gradio任务完成，并更新任务状态"""
    global jobs
    try:
        print(f"后台线程启动：正在监控 job_id: {job_id}")
        # result() 是一个阻塞调用，它会一直等到任务完成
        result_tuple = job.result()
        video_path = result_tuple[0]['video'] if result_tuple else None
        
        if video_path:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = {
                "video_path": video_path
            }
            print(f"任务成功 job_id: {job_id}, 结果: {video_path}")
        else:
            raise ValueError("Gradio任务完成，但没有返回有效的视频路径。")

    except Exception as e:
        print(f"任务失败 job_id: {job_id}, 错误: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)



@app.post("/generate_async")
async def generate_video(
# 这里定义所有API接收的参数
input_image: UploadFile = File(...),
prompt: str = Form(...),
total_second_length: int = Form(1),
end_frame: Optional[UploadFile] = File(None),
section_settings_json: str = Form("[]"),
seed: int = Form(141955780),
resolution: int = Form(640), # Gradio API文档显示为字符串
n_prompt: str = Form(""),
latent_window_size: int = Form(9),
steps: int = Form(25),
cfg: float = Form(1.0),
gs: float = Form(10.0),
rs: float = Form(0.0),
gpu_memory_preservation: float = Form(6.0),
use_teacache: bool = Form(True),
use_random_seed: bool = Form(False),
mp4_crf: int = Form(16),
all_padding_value: float = Form(1.0),
end_frame_strength: float = Form(1.0),
frame_size_setting: str = Form("1秒 (33幀)"), # 注意，您的文档是繁体，这里保持一致
keep_section_videos: bool = Form(False),
lora_scales_text: str = Form("0.8,0.8,0.8"),
output_dir: str = Form("outputs"),
save_section_frames: bool = Form(False),
use_all_padding: bool = Form(False),
use_lora: bool = Form(False),
lora_mode: str = Form("從目錄選擇"), # 注意繁体
lora_dropdown1: str = Form("無"), # 注意繁体
lora_dropdown2: str = Form("無"), # 注意繁体
lora_dropdown3: str = Form("無"), # 注意繁体
save_tensor_data: bool = Form(False),
fp8_optimization: bool = Form(True),
batch_count: int = Form(1),
frame_save_mode: str = Form("不保存"),
use_vae_cache: bool = Form(False),
use_queue: bool = Form(False),
save_settings_on_start: bool = Form(False),
alarm_on_completion: bool = Form(False),
):
    """
    接收参数并调用Gradio客户端来生成视频。
    """
    global gradio_client
    if not gradio_client:
        try:
            print(f"正在连接到Gradio服务: {GRADIO_APP_URL}...")
            gradio_client = GradioClient(GRADIO_APP_URL)
            print("Gradio服务连接成功！")
        except Exception as e:
            print(f"错误：无法连接到Gradio服务。请确保FramePack-eichi正在运行在 {GRADIO_APP_URL}。")
            gradio_client = None
            raise HTTPException(status_code=503, detail="Gradio服务未连接，请先启动FramePack-eichi应用。")
   
    temp_files = []
    try:
        input_image_path = save_upload_file_to_temp(input_image)
        temp_files.append(input_image_path)
        
        end_frame_path = save_upload_file_to_temp(end_frame) if end_frame else None
        if end_frame_path:
            temp_files.append(end_frame_path)

        # =============================================================================
        # === 关键修复：构建一个包含所有命名参数的字典 ===
        # =============================================================================
        
        params = {
            "input_image": handle_file(input_image_path),
            "prompt": prompt,
            "n_prompt": n_prompt,
            "seed": seed,
            "total_second_length": total_second_length,
            "latent_window_size": latent_window_size,
            "steps": steps,
            "cfg": cfg,
            "gs": gs,
            "rs": rs,
            "gpu_memory_preservation": gpu_memory_preservation,
            "use_teacache": use_teacache,
            "use_random_seed": use_random_seed,
            "mp4_crf": mp4_crf,
            "all_padding_value": all_padding_value,
            "end_frame": handle_file(end_frame_path) if end_frame_path else None,
            "end_frame_strength": end_frame_strength,
            "frame_size_setting": frame_size_setting,
            "keep_section_videos": keep_section_videos,
            "lora_files": None, # API不支持文件上传，设为None
            "lora_files2": None,
            "lora_files3": None,
            "lora_scales_text": lora_scales_text,
            "output_dir": output_dir,
            "save_section_frames": save_section_frames,
            # "section_settings" 似乎没有在API文档中暴露，暂时不传递
            "use_all_padding": use_all_padding,
            "use_lora": use_lora,
            "lora_mode": lora_mode,
            "lora_dropdown1": lora_dropdown1,
            "lora_dropdown2": lora_dropdown2,
            "lora_dropdown3": lora_dropdown3,
            "save_tensor_data": save_tensor_data,
            "tensor_data_input": None, # API不支持文件上传，设为None
            "fp8_optimization": fp8_optimization,
            "resolution": resolution,
            "batch_count": batch_count,
            "frame_save_mode": frame_save_mode,
            "use_vae_cache": use_vae_cache,
            "use_queue": use_queue,
            "prompt_queue_file": None, # API不支持文件上传，设为None
            "save_settings_on_start": save_settings_on_start,
            "alarm_on_completion": alarm_on_completion,
        }

        print("参数准备完毕，正在调用Gradio API...")
        
        # 使用命名参数和正确的API端点名称进行调用
        job = gradio_client.submit(
            **params,
            api_name="/validate_and_process" 
        )
        job_id = str(uuid.uuid4())
        global jobs
        jobs[job_id] = {
            "status": "running",
            "job_object": job, # 存储job对象以备将来使用（尽管这里没用上）
            "result": None,
            "error": None
        }

        # 创建并启动一个后台线程来等待任务完成
        thread = threading.Thread(target=watch_job_status, args=(job, job_id))
        thread.start()
        
        # 立即返回job_id，不等待任务完成
        return {"status": "running", "job_id": job_id}
    except Exception as e:
        print(f"提交任务时发生错误: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"提交任务时发生内部错误: {e}")
    finally:
        # 注意：临时文件不能在这里删除了，因为后台任务还在使用它们。
        # 需要一个更完善的清理机制，但为了简化，暂时不实现。
        pass

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """根据 job_id 查询任务状态和结果"""
    job_info = jobs.get(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    
    response = {"job_id": job_id, "status": job_info["status"]}
    
    if job_info["status"] == "completed":
        response["result"] = job_info["result"]
    elif job_info["status"] == "failed":
        response["error"] = job_info["error"]
        
    return response

@app.get("/download/{job_id}")
async def get_job_status(job_id: str):
    """根据 job_id 查询任务状态和结果"""
    job_info = jobs.get(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    
    # 3. 从文件名推断媒体类型
    real_path=job_info['result']['video_path']
    filename = os.path.basename(real_path)
    media_type = "video/mp4" # 默认为 mp4
        
    # 4. 使用 FileResponse 返回文件
    return FileResponse(path=real_path, media_type=media_type, filename=filename)

@app.get("/")
def read_root():
    return {"status": "FramePack-eichi API 正在运行"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002,timeout_keep_alive=7200)