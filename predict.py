from typing import Any
import io
import base64
from PIL import Image
import numpy as np
from cog import BasePredictor, Input, Path
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, list_dir, get_agnostic_mask_hd, get_agnostic_mask_dc
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

def pil_to_base64(image, format="PNG"):
    """
    Converts a PIL image to a base64 string.
    
    :param image: PIL Image object
    :param format: Format to save the image (default is PNG)
    :return: Base64 encoded string of the image
    """
    # Create a BytesIO object to hold the image data
    buffered = io.BytesIO()
    
    # Save the image to the buffer
    image.save(buffered, format=format)
    
    # Get the byte data from the buffer
    img_bytes = buffered.getvalue()
    
    # Encode the byte data to a base64 string
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    
    return img_base64

class Predictor(BasePredictor):
    def setup(self):
        self.densepose_predictor = DensePosePredictor(config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
                                                      weights_path="./ckpts/densepose/model_final_162be9.pkl")

        self.openpose = OpenPose(body_model_path="./ckpts/openpose/body_pose_model.pth")

        self.parsing = Parsing(atr_path="./ckpts/humanparsing/parsing_atr.onnx",
                               lip_path="./ckpts/humanparsing/parsing_lip.onnx")
        vt_model_hd = LeffaModel(pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
                                 pretrained_model="./ckpts/virtual_tryon.pth", dtype="float16")
        
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)

    # Define the arguments and types the model takes as input
    def predict(self, person_image: Path = Input(description="Image of the person"), garment_image: Path = Input(description="Image of the garment")) -> str:
        vt_garment_type = "upper_body"
        step=10
        ref_acceleration=True
        step=10
        scale=2.5
        seed=42
        vt_repaint=False
        src_image = Image.open(person_image)
        ref_image = Image.open(garment_image)
        
        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        src_image = src_image.convert("RGB")
        model_parse, _ = self.parsing(src_image.resize((384, 512)))
        keypoints = self.openpose(src_image.resize((384, 512)))
        mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)

        src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
        src_image_seg = Image.fromarray(src_image_seg_array)
        densepose = src_image_seg
             
        # Leffa
        transform = LeffaTransform()

        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)
         
        output = self.vt_inference_hd(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint,)
        gen_image = output["generated_image"][0]
        #gen_image.save("gen_image.png")
        return pil_to_base64(gen_image, format="PNG")
