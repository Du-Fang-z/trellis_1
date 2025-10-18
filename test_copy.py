import os
import imageio
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import uvicorn

# 环境变量设置
os.environ['SPCONV_ALGO'] = 'native'
# 初始化 FastAPI 应用
app = FastAPI()

# 加载模型
pipeline2 = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline2.cuda()

@app.post("/sketch2trellis/text/video")
async def generate_text_video(prompt: str = Form(...)):    
    # 运行 pipeline
    outputs = pipeline2.run(
        prompt,
        seed=1,
        sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
        slat_sampler_params={"steps": 12, "cfg_strength": 3},
    )

    # 渲染视频
    os.makedirs("results", exist_ok=True)
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    video_path = "results/asset_gs.mp4"
    imageio.mimsave(video_path, video, fps=30)

    # 返回文件作为响应
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename="asset_gs.mp4"
    )


if __name__ == "__main__":
    uvicorn.run("test_copy:app", host="0.0.0.0", port=8000, reload=True)
    