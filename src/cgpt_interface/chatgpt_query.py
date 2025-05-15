import base64

import openai
from src.loggers import save_images

# Prompts
positive_prompt = """
    I want images of stripes of a zebra. 
    I would like a close up of the stripes with different angles.
    """

for i in range(1):
    n = 1
    print(f"Generating batch {i}")
    response = openai.images.generate(
        model="gpt-image-1",
        prompt=positive_prompt,
        n=n,
        size="1024x1024",
        output_format="png",
        quality="low",
    )

    image_data = [base64.b64decode(response.data[i].b64_json) for i in range(n)]
    save_images(image_data, concept_name=f"zebra_stripes{i}")

print("All images generated and saved.")
