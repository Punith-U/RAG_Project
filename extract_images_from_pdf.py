import os
from pypdf import PdfReader
from PIL import Image
import io
import json

def extract_images_from_pdf(pdf_path):
    images = []
    reader = PdfReader(pdf_path)
    for page_num, page in enumerate(reader.pages):
        try:
            xobject = page["/Resources"]["/XObject"].get_object()
        except Exception:
            continue
        for obj_name in xobject:
            obj = xobject[obj_name].get_object()
            if obj.get("/Subtype") == "/Image":
                try:
                    data = obj._data
                    image = Image.open(io.BytesIO(data))
                    os.makedirs("images", exist_ok=True)
                    image_path = f"images/{os.path.basename(pdf_path)}_page{page_num+1}_{obj_name[1:]}.png"
                    image.save(image_path)
                    images.append({"path": image_path, "page": page_num+1, "source": pdf_path})
                except Exception:
                    continue
    return images

def save_image_mapping(images, path="image_mapping.json"):
    with open(path, "w") as f:
        json.dump(images, f, indent=2)
