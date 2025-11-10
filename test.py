import os
import imageio
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import uvicorn
import time

# 环境变量设置
os.environ['SPCONV_ALGO'] = 'native'
# 初始化 FastAPI 应用
app = FastAPI()

# 加载模型
pipeline1 = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline1.cpu()

@app.post("/sketch2trellis/picture23d/")
async def generate_3d(file: UploadFile = File(...)):    
    # 确保 temp 文件夹存在
    os.makedirs("temp", exist_ok=True)

    # 保存上传的图片
    input_path = f"temp/{file.filename}"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 加载图片
    image = Image.open(input_path)
    time1 = time.time()
    pipeline1.cuda()
    time2 = time.time()
    print(f"Pipeline moved to GPU in {time2 - time1:.2f} seconds.")
    # 运行 pipeline
    outputs = pipeline1.run(
        image,
        seed=1,
        sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
        slat_sampler_params={"steps": 12, "cfg_strength": 3},
    )
    # 渲染视频
    os.makedirs("results", exist_ok=True)
    video_paths = {}
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    video_path = "results/asset_gs.mp4"
    imageio.mimsave(video_path, video, fps=30)
    video_paths["gaussian"] = video_path

    # video = render_utils.render_video(outputs['mesh'][0])['normal']
    # video_path = "results/sample1_mesh.mp4"
    # imageio.mimsave(video_path, video, fps=30)
    # video_paths["mesh"] = video_path

    # 导出 GLB 文件
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=512,
    )
    glb_path = "results/asset.glb"
    glb.export(glb_path)
    time3 = time.time()
    print(f"3D model exported in {time3 - time2:.2f} seconds.")
    pipeline1.cpu()
    time4 = time.time()
    print(f"Pipeline moved back to CPU in {time4 - time3:.2f} seconds.")

    # # 保存 PLY 文件
    # ply_path = "results/sample1.ply"
    # outputs['gaussian'][0].save_ply(ply_path)

    # return {
    #     "gaussian_video": video_paths["gaussian"],
    #     # "mesh_video": video_paths["mesh"],
    #     # "glb_file": glb_path,
    #     "ply_file": ply_path,
    # }

    # 返回文件作为响应
    return FileResponse(
        path=glb_path,
        media_type="application/octet-stream",
        filename="asset.glb"
    )




@app.post("/sketch2trellis/video")
async def generate_3d_video(file: UploadFile = File(...)):    
    # 确保 temp 文件夹存在
    os.makedirs("temp", exist_ok=True)

    # 保存上传的图片
    input_path = f"temp/{file.filename}"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 加载图片
    image = Image.open(input_path)

    # 运行 pipeline
    outputs = pipeline1.run(
        image,
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
    uvicorn.run("test:app", host="0.0.0.0", port=8000, reload=True)
    