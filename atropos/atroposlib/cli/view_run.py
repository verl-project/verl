import argparse
import asyncio

import aiohttp
import gradio as gr
from transformers import AutoTokenizer

from atroposlib.utils.io import parse_http_response


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
                data = await parse_http_response(response)
                print(data)
                if data["batch"] is not None:
                    return data["batch"]
                await asyncio.sleep(1)


async def build_interface(group_size, max_token_len, tokenizer, port):
    async def grab_batch():
        tok = AutoTokenizer.from_pretrained(tokenizer)
        data = await check_for_batch()
        print(data)
        chats = [tok.decode(chat) for chat in data[0]["tokens"]]

        # find common prefix
        prefix = find_common_prefix(chats)
        return (
            (prefix,)
            + tuple([chat.split(prefix)[1] for chat in chats[:group_size]])
            + tuple(data[0]["scores"][:group_size])
        )

    with gr.Blocks() as demo:
        prefix_blk = gr.Textbox(label="Prefix")
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
            outputs=[prefix_blk] + outputs_blks + score_blks,
            api_name="get_batch",
        )
    await register_to_api(group_size, max_token_len)
    demo.launch(server_port=port, share=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-token-len", type=int, default=2048)
    parser.add_argument(
        "--tokenizer", type=str, default="NousResearch/DeepHermes-3-Llama-3-8B-Preview"
    )
    args = parser.parse_args()
    asyncio.run(
        build_interface(args.group_size, args.max_token_len, args.tokenizer, args.port)
    )


if __name__ == "__main__":
    main()
