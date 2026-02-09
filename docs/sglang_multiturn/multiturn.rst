Multi-turn Rollout Support
==========================

Last updated: 06/27/2025.

Basic Configuration
~~~~~~~~~~~~~~~~~~~

To enable multi-turn rollout, make sure to configure the following fields in your rollout configuration:

.. code-block:: yaml

    actor_rollout_ref: 
        rollout: 
            multi_turn: True
            name: "sglang"

These configuration activates the sglang engine for multi-turn interaction during rollout.

Custom Tool Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

For custom environment interaction tools, you can implement your own tools based on ``verl.tools.base_tool.BaseTool``. Then, specify your tool configurations in a YAML file:

.. code-block:: yaml

    tools:
      - class_name: ""
        config: 
            type: native
        tool_schema:

You may refer to GSM8KTool_example_configuration_, which is one example of the tool configurations. Its implementation can be found in gsm8k_tool.py_.

Finally, set the ``tools_config_file`` in your rollout config:

.. code-block:: yaml

    actor_rollout_ref:
        rollout:
            tool_kwargs:
                tools_config_file: <path_to_tool_yaml_file>

This allows integration of customized tool behaviors during actor rollout steps.

If you want rollout with simulated interaction, you can set the ``interaction_config_file`` in your rollout config:

.. code-block:: yaml

    interaction:
      - class_name: ""
        config: {}

.. code-block:: yaml

    actor_rollout_ref:
        rollout:
            interaction_config_file: <path_to_interaction_yaml_file>

If your tool creates multi-modal inputs, you should return a list of multi-modal inputs in your tool.execute() implementation.

Image and video should be processed before returning. For example, if you are using Qwen2.5-VL, you can use the following code to get the representations:

.. code-block:: python

    async def create(self, ...) -> tuple[str, ToolResponse]:
        ...
        from verl.utils.dataset.vision_utils import process_image, process_video

        img1 = process_image(img1)
        video1 = process_video(video1)

        # due to the (image | video) key is ("image" | "video") instead of ("images" | "videos") in vllm, we need to use ("image" | "video") to specify list of images/videos
        # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
        return instance_id, ToolResponse(image=[img1, ...], video=[video1, ...], text="...")

    async def execute(self, ...) -> Tuple[str | Dict[str, Any], float, dict]:
        ...
        from verl.utils.dataset.vision_utils import process_image, process_video

        img1 = process_image(img1)
        video1 = process_video(video1)

        # due to the (image | video) key is ("image" | "video") instead of ("images" | "videos") in vllm, we need to use ("image" | "video") to specify list of images/videos
        # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
        return ToolResponse(image=[img1, ...], video=[video1, ...], text="..."), 0, {}

remeber to set ``return_multi_modal_inputs: False`` in your dataset config in order to process the multi-modal inputs in the rollout correctly.
Refer to the `Handling Multi-Modal Inputs in Datasets`_ section for more details.

MCP Tool Configuration
~~~~~~~~~~~~~~~~~~~~~~

For MCP interaction tools, you can flexibly configure them using a YAML file. The typical setup is as follows:

.. code-block:: yaml

    tools:
      - class_name: ""
        config:
            type: mcp
        mcp:
            mcp_servers_config_path: ./mcp_server.json
            tool_selected_list: {}

The ``tool_selected_list`` field is optional and specifies which tools to use from the servers. If you want to enable all available tools, simply omit this attribute. Besides, ``mcp_servers_config_path`` points to a JSON file containing the MCP server configurations. For example:

.. code-block:: json

      {
          "mcpServers": {
              "SSE Server": {
                  "url": "your_server_url",
                  "auth_token": "your_server_api_token"
              },
              "STDIO Server": {
                  "command": "npx",
                  "args": ["-y", "server-mcp@0.2.1"],
                  "env": {
                    "SERVER_API_KEY": "your_server_api_token"
                  }
              }
          }
      }

Since the content formats returned by the MCP server may vary, users can inherit from ``MCPBaseTool`` and override the ``_parse_tool_result`` method to implement custom parsing logic.

.. code-block:: python

   class MCPYourTool(MCPBaseTool):
       def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
           super().__init__(config, tool_schema)

       def _parse_tool_result(self, content: list) -> Tuple[str, dict]:
           ...

Overall, you may refer to mcp_search_tool.py_ and mcp_tool_config.yaml_ for custom implementation and configuration.

Multi-turn Tokenization
~~~~~~~~~~~~~~~~~~~~~~~

Tokenizing multi-turn rollouts poses a challenge: after applying the chat template and tokenizing the full message list, it's hard to identify which tokens belong to assistant messages. Since the token list is flat, it lacks direct alignment with the message roles.

To address this, we adopt a **delta-based tokenization** strategy. Each time the LLM generates a new message, we:

1. Apply the chat template to all prior messages (`messages[:i]`).
2. Apply the chat template again including the latest message (`messages[:i+1]`).
3. Tokenize only the *delta* between these two serialized message strings.

This ensures that only tokens generated by the assistant are included in the loss mask.

.. code-block:: python

   # When using tokenizer
   # Exclude the assistant prompt (e.g., "<|im_start|>assistant") from the loss by setting add_generation_prompt=True
   prev = tokenizer.apply_chat_template(messages[:i], add_generation_prompt=True, tokenize=False)
   curr = tokenizer.apply_chat_template(messages[:i+1], add_generation_prompt=False, tokenize=False)
   token_ids += tokenizer.encode(curr[len(prev):], add_special_tokens=False)
   loss_mask += [1] * len(token_ids)  # Mask only the new assistant tokens

.. code-block:: python

   # When using processor
   # Exclude the assistant prompt (e.g., "<|im_start|>assistant") from the loss by setting add_generation_prompt=True
   prev = processor.apply_chat_template(messages[:i], add_generation_prompt=True, tokenize=False)
   prev_model_inputs = processor(text=prev, images=images, videos=videos, return_tensors="pt")[0].tolist()
   curr = processor.apply_chat_template(messages[:i+1], add_generation_prompt=False, tokenize=False)
   curr_model_inputs = processor(text=curr, images=images, videos=videos, return_tensors="pt")[0].tolist()
   token_ids += curr_model_inputs["input_ids"][len(prev_model_inputs["input_ids"]):]
   loss_mask += [1] * len(token_ids)  # Mask only the new assistant tokens

While we've validated this produces consistent results with full message tokenization, future models' chat template could break compatibility. To guard against silent inconsistencies, we compare the delta-based tokenization with full-tokenization results by default at the end of each rollout.

If you see the following warning, you can check the mismatched substring in the log:

.. code-block::

    Inconsistent training and inference tokenization detected. This may lead to unexpected behavior during training. Please review your chat template to determine if this is intentional. For more information, refer to the multiturn README.md.

The tokenization sanity check mode can be configured using the ``actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode`` parameter, which accepts the following values:

- ``strict`` (default): Performs strict comparison between delta-based and full tokenization results, raising warnings for any differences.

- ``ignore_strippable``: Ignores differences in whitespace characters (``\n``, ``\t``, ``\r``, spaces) while still checking for meaningful text mismatches. This is useful when debugging chat template issues where whitespace variations are expected and acceptable.

- ``disable``: Completely disables the tokenization sanity check. Only use this if you have thoroughly validated that tokenization discrepancies are expected and won't impact training.

Example configuration:

.. code-block:: yaml

    actor_rollout_ref:
        rollout:
            multi_turn:
                tokenization_sanity_check_mode: "ignore_strippable"  # Choose from: "disable", "ignore_strippable", "strict"

Handling Multi-Modal Inputs in Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your dataset includes multi-modal inputs (such as images or videos), you can control whether these are pre-processed and included in each sample by setting the return_multi_modal_inputs flag in your dataset config (used by RLHFDataset).

- ``return_multi_modal_inputs: True`` (default): The dataset will pre-process and include a multi_modal_inputs dictionary for each sample. This dict contains the model-ready representations (e.g., image tensors, video tensors, etc.) as produced by your processor. This is useful for single-turn or SFT-style training, where the model expects all modalities to be present in the batch.

- ``return_multi_modal_inputs: False``: The dataset will not include the multi_modal_inputs field. This is recommended for multi-turn RL or tool-augmented rollouts, where the model may generate new multi-modal inputs dynamically during rollout, and you want to avoid conflicts or redundant data in the batch.


Special Cases
^^^^^^^^^^^^^

Some models (e.g., Qwen/QwQ-32B and Qwen3 series) remove internal reasoning content during chat template rendering. As a result, the message content can vary across turns, making the delta-based tokenization inaccurate.

For example, for the following conversation:

.. code-block:: python

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "<think>user asked about a simple math question.</think> 2 + 2 = 4."},
        {"role": "user", "content": "Explain why."},
        {"role": "assistant", "content": "<think>user wants to know the reasoning behind the answer. Search for a good explanation</think>",
         "tool_calls": [{"id": "tool1", "type": "search", "arguments": {"query": "Why is 2 + 2 = 4?"}}]},
        {"role": "tool", "content": "The sum of two and two is four because it is a basic arithmetic operation."},
        {"role": "assistant", "content": "<think>The tool provided a good explanation.</think>The sum of two and two is four because it is a basic arithmetic operation."}
    ]

1. Qwen/QwQ-32B will remove all reasoning content except the last assistant message after applying the chat template.

.. code-block:: text

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    What is 2 + 2?<|im_end|>
    <|im_start|>assistant
     2 + 2 = 4.<|im_end|>
    <|im_start|>user
    Explain why.<|im_end|>
    <|im_start|>assistant
    <tool_call>
    {"name": "", "arguments": {"query": "Why is 2 + 2 = 4?"}}
    </tool_call><|im_end|>
    <|im_start|>user
    <tool_response>
    The sum of two and two is four because it is a basic arithmetic operation.
    </tool_response><|im_end|>
    <|im_start|>assistant
    <think>The tool provided a good explanation.</think> The sum of two and two is four because it is a basic arithmetic operation.<|im_end|>

2. Qwen3 series will remove all reasoning content before the last user message.

.. code-block:: text

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    What is 2 + 2?<|im_end|>
    <|im_start|>assistant
     2 + 2 = 4.<|im_end|>
    <|im_start|>user
    Explain why.<|im_end|>
    <|im_start|>assistant
    <think>
    user wants to know the reasoning behind the answer. Search for a good explanation
    </think>

    <tool_call>
    {"name": "", "arguments": {"query": "Why is 2 + 2 = 4?"}}
    </tool_call><|im_end|>
    <|im_start|>user
    <tool_response>
    The sum of two and two is four because it is a basic arithmetic operation.
    </tool_response><|im_end|>
    <|im_start|>assistant
    <think>
    The tool provided a good explanation.
    </think>

    The sum of two and two is four because it is a basic arithmetic operation.<|im_end|>

To handle this, we fall back to a **fixed base conversation** containing only a single system and user message. Since this base doesn't include assistant messages or reasoning content, it remains consistent across turns.

.. code-block:: python

    BASE_CHAT_HISTORY = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I am a user."}
    ]
    prev = tokenizer.apply_chat_template(BASE_CHAT_HISTORY, add_generation_prompt=True, tokenize=False)
    curr = tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, messages[i]], add_generation_prompt=False, tokenize=False)
    token_ids += tokenizer.encode(curr[len(prev):], add_special_tokens=False)
    loss_mask += [1] * len(token_ids)

This method works well for Qwen3 series. However, Qwen/QwQ-32B currently has a bug in its chat template. A fix_ has been proposed but not yet adopted. Until then, use the following command to download the fixed model revision:

.. code-block:: bash

    pip install huggingface_hub
    hf download Qwen/QwQ-32B --revision refs/pr/81

.. _fix: https://huggingface.co/Qwen/QwQ-32B/discussions/81

Discrepancy Between Training and Inference Templates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although the above approach fixes the delta mismatch issue, the removal of reasoning content in the inference-time chat template introduces a new discrepancy: training uses the full reasoning content, while inference does not.

This mismatch can affect model performance in unpredictable ways. To avoid it, we default to using the full response (including reasoning) for both training and rollout.

However, this approach comes with trade-offs:

1. Long reasoning contents can easily exceed the model's context window, especially in multi-turn rollout.
2. There's a mismatch between rollout and production environment now—models will not have reasoning content from past turns if you use the default chat template in production.

We are still evaluating the impact of these issues. If you experience context length problems or prefer rollouts that match production (i.e., exclude reasoning), you can enable:

``actor_rollout_ref.rollout.multi_turn.use_inference_chat_template = True``

GSM8K Multi-turn Training Performance  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the training performance of multi-turn rollout on the GSM8K task HERE_.

.. _HERE: https://wandb.ai/zhaochenyang20/gsm8k_async_rl/runs/1ro1r7om?nw=nwuserzhaochenyang20

.. _GSM8KTool_example_configuration: https://github.com/volcengine/verl/blob/main/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml

.. _gsm8k_tool.py: https://github.com/volcengine/verl/blob/main/verl/tools/gsm8k_tool.py

.. _mcp_search_tool.py: https://github.com/volcengine/verl/blob/main/verl/tools/mcp_search_tool.py

.. _mcp_tool_config.yaml: https://github.com/volcengine/verl/blob/main/examples/sglang_multiturn/config/tool_config/mcp_tool_config.yaml

Interaction System
~~~~~~~~~~~~~~~~~~

For dynamic conversational feedback during RL training, see:

.. toctree::
   :maxdepth: 1

   interaction_system

Search Tool Integration
~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   search_tool_example

Code Walkthrough
~~~~~~~~~~~~~~~~~~~~~~~
If you want to learn more in depth about the code execution flow, please read https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/rlhf/verl/multi-turn/code-walk-through
