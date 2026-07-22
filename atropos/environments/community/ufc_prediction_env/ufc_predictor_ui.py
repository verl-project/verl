import base64
import os
from io import BytesIO

import openai
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from PIL import Image

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def process_image(image_file):
    """Convert uploaded image to base64"""
    img = Image.open(image_file)
    # Convert RGBA to RGB if necessary
    if img.mode == "RGBA":
        img = img.convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.route("/")
def home():
    return render_template("predictor.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get uploaded images
        red_fighter = request.files["red_fighter"]
        blue_fighter = request.files["blue_fighter"]

        if not red_fighter or not blue_fighter:
            return jsonify({"error": "Please upload both fighter images"}), 400

        # Process images to base64
        red_image = process_image(red_fighter)
        blue_image = process_image(blue_fighter)

        # Create the prompt
        prompt_text = (
            "🎤 LADIES AND GENTLEMEN! Welcome to the most electrifying show in sports entertainment "
            "Let's break down this matchup that's got everyone talking!\n\n"
            "In the red corner, we have:(YOUR FIRST IMAGE):\n"
            "And in the blue corner: (YOUR SECOND IMAGE):\n\n"
            "Now, as your favorite fight comentator, I want you to:\n"
            "create a fight commentary of whats happening in the fight live\n"
            "Give us your best fight commentary! Make it exciting, make it dramatic, "
            "make it sound like you're calling the fight live! "
            "Throw in some classic commentator phrases, maybe a 'OH MY GOODNESS!' or two, "
            "and definitely some dramatic pauses for effect.\n\n"
            "End your masterpiece with your prediction in this exact format:\n"
            "\\boxed{Red} or \\boxed{Blue}"
            "PLEASE FORMAT THE COMMENTARY IN THE EXACT FORMAT AS THE EXAMPLE BELOW:\n"
            "[S1]Hello im your host  [S2] And so am i (name) [S1] Wow. Amazing. (laughs) "
            "[S2] Lets get started! (coughs) ( add lots of coughs and laughs)\n\n"
        )

        # Create the messages for the API call
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{red_image}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{blue_image}"},
                    },
                ],
            }
        ]

        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.95,
        )

        # Extract the prediction
        prediction = response.choices[0].message.content

        return jsonify({"prediction": prediction, "success": True})

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


if __name__ == "__main__":
    app.run(debug=True)
