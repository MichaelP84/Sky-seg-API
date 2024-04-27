import torch
from torch.cuda import amp
import numpy as np
import os
import cv2
from train import AnimeSegmentation
import modal
from modal import Image, web_endpoint
import base64


net_name = 'isnet_is'
ckpt = 'ckpt/isnetis.ckpt'
im_size = 1024
# device = 'cuda:0'
device = 'cpu'
fp32 = False
only_matted = False

device = torch.device(device)
model = AnimeSegmentation.try_load(net_name, ckpt, device, im_size)
model.eval()
model.to(device)

seg_image = Image.debian_slim().copy_local_dir(
    "/ckpt"
).pip_install(
    "numpy==1.24.3",
    "opencv-python-headless",
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "scipy",
    "torch",
    "torchvision",
    "pytorch-lightning", 
    "pillow", 
    "kornia", 
    "timm", 
    "gradio"
)

# copy model checkpoint to Image
stub = modal.Stub("bezier")


def get_mask(model, input_img, use_amp=True, s=640):
    input_img = (input_img / 255).astype(np.float32)
    h, w = h0, w0 = input_img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    tmpImg = torch.from_numpy(img_input).type(torch.FloatTensor).to(model.device)
    with torch.no_grad():
        if use_amp:
            with amp.autocast():
                pred = model(tmpImg)
            pred = pred.to(dtype=torch.float32)
        else:
            pred = model(tmpImg)
        pred = pred.cpu().numpy()[0]
        pred = np.transpose(pred, (1, 2, 0))
        pred = pred[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        pred = cv2.resize(pred, (w0, h0))[:, :, np.newaxis]
        return pred

@stub.function(
    image=seg_image,
    container_idle_timeout=120,
)
@web_endpoint(method="POST")
def process_mask(file: dict):
    base64_image_data = file["image"]
    if "," in base64_image_data:
        base64_image_data = base64_image_data.split(",")[1]
        
    image_data = base64.b64decode(base64_image_data)
    decoded = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    mask = get_mask(model, img, use_amp=not fp32, s=im_size)
    
    img = np.concatenate((mask * img + 1 - mask, mask * 255), axis=2).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    
    # convert to base64
    retval, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {"mask": img_base64}
    
# @web_endpoint(method="POST")
# def process_mask(file):
#     # base64_image_data = file["image"]
#     # if "," in base64_image_data:
#     #     base64_image_data = base64_image_data.split(",")[1]
        
#     # image_data = base64.b64decode(base64_image_data)
#     # print(image_data)
#     # decoded = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
#     img = cv2.cvtColor(cv2.imread(file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#     mask = get_mask(model, img, use_amp=not fp32, s=im_size)
    
#     img = np.concatenate((mask * img + 1 - mask, mask * 255), axis=2).astype(np.uint8)
#     img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
#     cv2.imwrite(f'./out.png', img)
    
    
# def main():
#     process_mask("/Users/michaelpasala/Projects/Toona/refs/group.png")
    
# if __name__ == "__main__":
#     main()