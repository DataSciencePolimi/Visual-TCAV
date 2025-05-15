import base64
from pathlib import Path

import openai

from src.generative_cav.loggers import save_images

# Input and output folders
input_dir = Path("data/test_images/zebra")

# Prompt
edit_prompt = "Remove the stripes from the Zebra in the image. If there is more than 1 zebra, remove the stripes from all of them."

# Process each image
image_data = []
for file in input_dir.glob("*.*"):
    if file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
        continue

    print(f"Processing: {file.name}")

    # Load image bytes
    with open(file, "rb") as f:
        # Use the `images.edit` API
        response = openai.images.edit(
            model="gpt-image-1",
            image=f,
            prompt=edit_prompt,
            n=1,
            size="1024x1024",
            quality="low",
        )

    result_b64 = response.data[0].b64_json
    result_bytes = base64.b64decode(result_b64)

    image_data.append(result_bytes)

save_images(image_data, concept_name=f"zebra_stripes_removed")
