import numpy as np
import torch
import cv2

from huggingface_hub import PyTorchModelHubMixin

from infer_ddcolor_colorization.ddcolor.basicsr.archs.ddcolor_arch import DDColor
from infer_ddcolor_colorization.ddcolor.inference.colorization_pipeline import ImageColorizationPipeline


MODEL_NAMES = ["ddcolor_paper", "ddcolor_paper_tiny", "ddcolor_modelscope", "ddcolor_artistic"]


class DDColorHF(DDColor, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__(**config)


class ImageColorizationPipelineHF(ImageColorizationPipeline):
    def __init__(self, model, input_size, device):
        self.device = device
        self.input_size = input_size
        self.model = model.to(self.device)
        self.model.eval()


class InferDDColor:
    def __init__(self, model_folder: str):
        self.model_name = ""
        self.colorizer = None
        self.model = None
        self.input_size = 512
        self.model_folder = model_folder
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def set_parameters(self, model_name: str = "ddcolor_paper", input_size: int = 512, cuda: bool = True):
        update_colorizer = self.colorizer is None

        if input_size % 32 != 0:
            raise ValueError("Invalid parameters: input size must be multiple of 32.")

        if self.model is None or self.model_name != model_name:
            self.model = DDColorHF.from_pretrained(f"piddnad/{model_name}",
                                                   cache_dir=self.model_folder)
            self.model_name = model_name
            update_colorizer = True

        if self.input_size != input_size:
            self.input_size = input_size
            update_colorizer = True

        device = torch.device("cuda") if cuda and torch.cuda.is_available() else torch.device("cpu")
        if self.device != device:
            self.device = device
            update_colorizer = True

        if update_colorizer:
            self.colorizer = ImageColorizationPipelineHF(model=self.model,
                                                         input_size=self.input_size,
                                                         device=self.device)

    def run(self, src_image: np.ndarray):
        channels = src_image.shape[-1] if src_image.ndim > 2 else 1
        if channels == 1:
            img = cv2.cvtColor(src_image, cv2.COLOR_GRAY2RGB)
        elif channels == 4:
            img = cv2.cvtColor(src_image, cv2.COLOR_RGBA2RGB)
        else:
            img = src_image

        return cv2.cvtColor(self.colorizer.process(img), cv2.COLOR_BGR2RGB)
