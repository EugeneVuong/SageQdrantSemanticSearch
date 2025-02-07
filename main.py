from waggle.plugin import Plugin
from waggle.data.vision import Camera
import yolov5
from sentence_transformers import SentenceTransformer
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from io import BytesIO
import base64
from langchain_sambanova import ChatSambaNovaCloud
from fastembed import TextEmbedding
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

# Converts PIL Image to Base64
def pil_image_to_base64(pil_image):
    byte_io = BytesIO()
    pil_image.save(byte_io, format='JPEG')
    byte_image = byte_io.getvalue()
    base64_image = base64.b64encode(byte_image).decode('utf-8')
    return base64_image

# Detects weapons in the Image
def detect_weapons(image, model):
    results = model(image, size=640)
    return results

# Gets the YOLOv5 Model
def get_model():
    model = yolov5.load("best.pt")
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image
    return model

def main():

    # Load the YOLOv5 Model
    model = get_model()
    # Image to Embedding
    clip = SentenceTransformer('clip-ViT-L-14')

    # Text to Embedding
    text_embed = TextEmbedding(model_name="snowflake/snowflake-arctic-embed-l")

    with Plugin() as plugin, Camera() as camera:
        for snapshot in camera.stream():
            results = detect_weapons(snapshot.data, model)

            predictions = results.pred[0]
            # boxes = predictions[:, :4] # x1, y1, x2, y2
            # scores = predictions[:, 4]
            categories = predictions[:, 5]

            print(len(categories))

            if len(categories) > 0:
                print("Found a weapon!")
                snapshot.save("weapon.jpg")
                
                # Image Vector Embedding
                image_embedding = clip.encode(Image.open('weapon.jpg'))
                
                # Multimodal LLM (Pretend It Is A Local Multimodal LLM on the Edge)
                multimodal_llm = ChatSambaNovaCloud(
                    model="Llama-3.2-11B-Vision-Instruct",
                    max_tokens=1024,
                    temperature=0.7,
                    top_p=0.01,
                )
                # Caption-Text Embedding
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe the image in detail, including objects, people, background, and any notable elements. Pay special attention to any weapons present—identify their type, appearance, and how they are being held or used. Include colors, textures, expressions, and any actions taking place to provide a comprehensive description."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{pil_image_to_base64(Image.open('weapon.jpg'))}"},
                        },
                    ],
                )

                # Text Embedding
                response = multimodal_llm.invoke([message])
                text_embedding = text_embed.embed([response.content])
                text_embedding = (list(text_embedding)[0]).tolist()
                image_embedding = image_embedding.tolist()

                plugin.upload_file("weapon.jpg", timestamp=snapshot.timestamp)
                plugin.publish("base64_image", pil_image_to_base64(Image.open('weapon.jpg')), timestamp=snapshot.timestamp)
                plugin.publish("description", response.content, timestamp=snapshot.timestamp)
                # Convert Embeddings to Strings
                plugin.publish("text_embedding", str(text_embedding), timestamp=snapshot.timestamp)
                plugin.publish("image_embedding",str(image_embedding), timestamp=snapshot.timestamp)

if __name__ == "__main__":
    main()
