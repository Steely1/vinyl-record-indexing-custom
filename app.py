import cv2
import torch
from PIL import Image

import mobileclip
import os
from collections import deque
from anthropic import Anthropic

import base64
import concurrent.futures
import csv
import datetime

client = Anthropic(api_key=os.environ['API_KEY'])

results = []

labels = ["vinyl record", "something else", "open palm"]

model, _, preprocess = mobileclip.create_model_and_transforms(
    "mobileclip_s0", pretrained="../ml-mobileclip/checkpoints/mobileclip_s0.pt"
)
tokenizer = mobileclip.get_tokenizer("mobileclip_s0")

text = tokenizer(labels)

BUFFER_MAX_LEN = 50
TO_QUALIFY = 0.2
FRAME_PERCENT = BUFFER_MAX_LEN * TO_QUALIFY
label_buffer = deque(maxlen=BUFFER_MAX_LEN)
recorded_vinyl_vectors = []
vinyl_count = 0
BREAK_PROMPT = "open palm"
BREAK_PROMPT_BUFFER_SIZE = 10
now = datetime.datetime.now()
formatted_date = now.strftime("%Y-%m-%d_%H:%M:%S")
NOT_FOUND_STR = "(not found)"

if not os.path.exists("vinyls"):
    os.makedirs("vinyls")


def embedding_has_not_been_recorded(embedding):
    # return False if embedding is not 80% cosine sim to any other recorded vinyl
    for recorded_vinyl in recorded_vinyl_vectors:
        if 100.0 * embedding @ recorded_vinyl.T > 50:
            return False

    return True


with torch.no_grad(), torch.cuda.amp.autocast():
    webcam = cv2.VideoCapture(0)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    while True:
        ret, frame = webcam.read()

        if not ret:
            print("Failed to grab frame")
            break

        image = preprocess(Image.fromarray(frame)).unsqueeze(0)

        image_features = model.encode_image(image)

        buffer_count = label_buffer.count("vinyl record")

        if buffer_count > FRAME_PERCENT and embedding_has_not_been_recorded(
            image_features
        ):
            vinyl_count += 1
            cv2.imwrite(f"vinyls/{formatted_date}_vinyl_{vinyl_count}.jpg", frame)
            label_buffer = []
            recorded_vinyl_vectors.append(image_features)
            print(f"Recorded vinyl {vinyl_count}")

        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        top_result = labels[torch.argmax(text_probs)]

        label_buffer.append(top_result)

        if label_buffer.count(BREAK_PROMPT) > BREAK_PROMPT_BUFFER_SIZE:
            break

        frame = cv2.putText(
            frame,
            top_result,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Vinyls recorded: {vinyl_count}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            webcam.release()
            break


def get_image_data(image_path):
    with open(image_path, "rb") as image_file:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_file,
                            },
                        },
                        {
                            "type": "text",
                            "text": """What vinyl record is in this image, and what genre is it? Limit your response to these three lines:
                            Artist: artist
                            Album Name: name
                            Genre: genre
                            """,
                        }
                    ],
                }
            ],
        )
        result = response.content[0].text
        if result.startswith("Artist"): 
            # print(result)
            artist = result.split("\n")[0].split(":")[1].strip()
            album = result.split("\n")[1].split(":")[1].strip()
            genre = result.split("\n")[2].split(":")[1].strip()
        else:
            artist = NOT_FOUND_STR
            album = NOT_FOUND_STR
            genre = NOT_FOUND_STR
            
        return {"artist": artist, "album": album, "AI-genre": genre, "my-category": "", "price-paid": "", "image-path": image_path}


with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(get_image_data, f"vinyls/{formatted_date}_vinyl_{i}.jpg"): i
        for i in range(1, vinyl_count + 1)
    }
    for future in concurrent.futures.as_completed(futures):
        i = futures[future]
        the_result = future.result()
        if not the_result["album"] == NOT_FOUND_STR:
            results.append(the_result)


with open("collection.csv", "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["artist", "album", "AI-genre", "my-category", "price-paid", "image-path"], delimiter='\t')
    writer.writeheader()
    writer.writerows(results)
