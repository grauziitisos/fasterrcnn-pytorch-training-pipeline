import numpy as np
import cv2
import torch
import torchvision
import glob as glob
import os
import time
import argparse
import yaml
import matplotlib.pyplot as plt
from models.create_fasterrcnn_model import create_model
from utils.annotations import (
inference_annotations, convert_detections
)
from utils.general import set_infer_dir
from utils.transforms import infer_transforms, resize
from utils.logging import log_to_json
from torch.utils.mobile_optimizer import optimize_for_mobile
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load("../../4AI_tikli_pet/2024-01-14/sign_mini_darknet/best_model.pth", map_location=DEVICE)

NUM_CLASSES = checkpoint['data']['NC']
CLASSES = checkpoint['data']['CLASSES']
build_model = create_model["fasterrcnn_mini_darknet"]
model = build_model(num_classes=NUM_CLASSES, coco_model=False)
model.load_state_dict(checkpoint['model_state_dict'])
torchscript_model = torch.jit.script(model)
torchscript_model_optimized = optimize_for_mobile(torchscript_model)
torch.jit.save(torchscript_model_optimized, "2024-01-14_sign_mini_darknet.pt")