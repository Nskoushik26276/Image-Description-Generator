!pip install transformers accelerate ipywidgets 'git+https://github.com/facebookresearch/detectron2.git' --quiet

from google.colab import files
from PIL import Image
import torch
from transformers import pipeline
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
import cv2
import numpy as np
import io
from google.colab.patches import cv2_imshow

setup_logger()

print("Please upload an image.")
uploaded = files.upload()

img_path = next(iter(uploaded))
print(f'User uploaded file "{img_path}" with length {len(uploaded[img_path])} bytes')

try:
    img = Image.open(io.BytesIO(uploaded[img_path])).convert('RGB')
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    print("Image loaded successfully.")
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

print("\nGenerating a general caption for the image...")
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=0 if torch.cuda.is_available() else -1)
inputs = captioner.preprocess(img)
outputs = captioner.model.generate(**{k: v.to(captioner.model.device) for k, v in inputs.items()}, max_length=40)
general_caption = captioner.postprocess(outputs)[0]['generated_text']
print(f"General Caption: {general_caption}")

print("\nDetecting objects in the image...")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(img_cv2)
instances = outputs["instances"]

v = Visualizer(img_cv2[:, :, ::-1], metadata=predictor.metadata, scale=0.8)
out = v.draw_instance_predictions(instances.to("cpu"))

cv2_imshow(out.get_image()[:, :, ::-1])

if len(instances) > 0:
    class_names = predictor.metadata.thing_classes
    object_list = [(i, class_names[instances.pred_classes[i].item()], instances.scores[i].item()) for i in range(len(instances))]

    print("\nDetected Objects:")
    for i, name, score in object_list:
        print(f"{i + 1}: {name} (Confidence: {score:.2f})")
else:
    print("No objects detected in the image.")
