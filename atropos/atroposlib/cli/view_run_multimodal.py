import argparse
import asyncio
import base64
import re
from io import BytesIO

import aiohttp
import gradio as gr
import PIL.Image
from transformers import AutoTokenizer


def find_common_prefix(strings):
    if not strings:
        return ""

    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


async def register_to_api(group_size, max_token_len):
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/reset_data") as response:
            print(await response.text())
        print(group_size)
        async with session.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "test",
                "wandb_project": "test",
                "batch_size": group_size
                * 8,  # * 8 just in case you want to just sample from a large group
                "max_token_len": max_token_len,
                "checkpoint_dir": "checkpoints",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 69,
            },
        ) as response:
            print("output of register is")
            print(await response.text())


async def check_for_batch():
    while True:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/batch") as response:
                data = await response.json()
                print(data)
                if data["batch"] is not None:
                    return data["batch"]
                await asyncio.sleep(1)


def extract_image_from_chat(chat_text):
    # Extract the base64 image data from the chat text
    # Support both jpeg and png formats
    image_pattern = r'data:image/(jpeg|png);base64,([^"\\]*)'
    match = re.search(image_pattern, chat_text)

    if match:
        base64_data = match.group(2)
        try:
            image_data = base64.b64decode(base64_data)
            image = PIL.Image.open(BytesIO(image_data))
            return image
        except Exception as e:
            print(f"Error decoding image: {e}")
    return None


def extract_text_from_chat(chat_text):
    # Try to extract text from JSON format first
    # Check if this is JSON multimodal content
    if '"type": "text"' in chat_text:
        text_pattern = r'"type": "text", "text": "([^"]*)"'
        match = re.search(text_pattern, chat_text)
        if match:
            return match.group(1)

    # If not in JSON format, look for [Image] prefix
    if "[Image]" in chat_text:
        return chat_text.split("[Image]", 1)[1].strip()

    # Return original text if no pattern is found
    return chat_text


async def build_interface(group_size, max_token_len, tokenizer, port):
    async def grab_batch():
        tok = AutoTokenizer.from_pretrained(tokenizer)
        data = await check_for_batch()
        print(data)
        chats = [tok.decode(chat) for chat in data[0]["tokens"]]

        # Find common prefix
        prefix = find_common_prefix(chats)

        # Handle base64 encoded image
        try:
            if "images" in data[0] and data[0]["images"] and data[0]["images"][0]:
                print("Found image data in batch")
                # Convert base64 string to image
                base64_image = data[0]["images"][0]

                # If it's already a PIL Image, use it directly
                if isinstance(base64_image, PIL.Image.Image):
                    image = base64_image
                # If it's a base64 string, decode it
                elif isinstance(base64_image, str):
                    # Remove data:image prefix if present
                    if base64_image.startswith("data:image"):
                        # Extract just the base64 part
                        image_data = base64_image.split(",", 1)[1]
                    else:
                        image_data = base64_image

                    # Decode base64 to bytes and create image
                    image_bytes = base64.b64decode(image_data)
                    image = PIL.Image.open(BytesIO(image_bytes))
                else:
                    print(f"Image type not recognized: {type(base64_image)}")
                    image = None
            else:
                # Try to extract image from chat text as fallback
                print("No images field found, trying to extract from chat text")
                image = extract_image_from_chat(prefix)
        except Exception as e:
            print(f"Error processing image: {e}")
            image = None

        # Extract text prompt from prefix
        text_prompt = extract_text_from_chat(prefix)

        return (
            image,  # Image
            text_prompt,  # Text prompt
            *[chat.split(prefix)[1] for chat in chats[:group_size]],  # Model outputs
            *data[0]["scores"][:group_size],  # Scores
        )

    with gr.Blocks() as demo:
        image_blk = gr.Image(label="Image", type="pil")
        prompt_blk = gr.Textbox(label="Text Prompt")

        with gr.Row():
            score_blks = [gr.Textbox(label=f"Score_{i+1}") for i in range(group_size)]

        with gr.Row():
            outputs_blks = [
                gr.Textbox(label=f"Output_{i+1}") for i in range(group_size)
            ]

        with gr.Row():
            grab_next = gr.Button(value="Grab Next Batch")

        grab_next.click(
            fn=grab_batch,
            outputs=[image_blk, prompt_blk] + outputs_blks + score_blks,
            api_name="get_batch",
        )
    await register_to_api(group_size, max_token_len)
    demo.launch(server_port=port, share=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-token-len", type=int, default=2048)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    args = parser.parse_args()
    asyncio.run(
        build_interface(args.group_size, args.max_token_len, args.tokenizer, args.port)
    )


if __name__ == "__main__":
    main()
