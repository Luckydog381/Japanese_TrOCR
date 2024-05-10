import numpy as np
from transformers import AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
import re
import jaconv

#load model
model_path = "Japanese_TrOCR/model/"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = VisionEncoderDecoderModel.from_pretrained(model_path)


def post_process(text):
    text = ''.join(text.split())
    text = text.replace('…', '...')
    text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
    text = jaconv.h2z(text, ascii=True, digit=True)
    return text

def infer(image):
    image = image.convert('L').convert('RGB')
    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
    ouput = model.generate(pixel_values)[0]
    text = tokenizer.decode(ouput, skip_special_tokens=True)
    text = post_process(text)
    return text

