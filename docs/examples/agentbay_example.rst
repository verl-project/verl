AgentBay Sandbox Integration
============================

Introduction
------------

`AgentBay <https://agentbay.console.aliyun.com>`_ is a cloud-based secure sandbox service provided by Alibaba Cloud Wuying.
It allows LLMs to execute code in isolated cloud environments during RL training,
enabling agentic RL workflows where models learn to write and run code to solve problems.

This integration provides ``AgentBayTool``, a tool that plugs into verl's multi-turn rollout
pipeline, giving the model access to a ``code_interpreter`` tool backed by AgentBay cloud sandbox.

Prerequisites
-------------

**1. Install the SDK**

.. code-block:: bash

   pip install wuying-agentbay-sdk

**2. Obtain an API key**

- Visit the `AgentBay Console <https://agentbay.console.aliyun.com/service-management>`_
- Create a service and obtain your API key
- Set it as an environment variable:

.. code-block:: bash

   export AGENTBAY_API_KEY=your_api_key_here

Alternatively, you can specify the API key in the tool configuration:

.. code-block:: yaml

   tools:
     - class_name: "verl.tools.agentbay_tool.AgentBayTool"
       config:
         api_key: "your_api_key_here"
         image_id: "code_latest"
         default_language: "python"
         type: native

Quick Start
-----------

**Step 1: Prepare the dataset**

We use GSM8K as an example, preprocessing it for code-interpreter style tool use:

.. code-block:: bash

   python examples/data_preprocess/gsm8k_agentbay.py \
       --local_save_dir ~/data/gsm8k_agentbay

This creates prompts that instruct the model to solve math problems by writing Python code
and executing it via the ``code_interpreter`` tool.

**Step 2: Run the training script**

.. code-block:: bash

   bash examples/sglang_multiturn/run_qwen0.5b_gsm8k_agentbay_1gpu.sh

This runs Qwen2.5-0.5B with GRPO on a single GPU, using AgentBay for code execution.
The script is configured for minimal resource usage (pipeline verification).

Tool Configuration
------------------

The AgentBay tool configuration is at
``examples/sglang_multiturn/config/tool_config/agentbay_tool_config.yaml``:

.. code-block:: yaml

   tools:
     - class_name: "verl.tools.agentbay_tool.AgentBayTool"
       config:
         image_id: "code_latest"
         default_language: "python"
         type: native
       tool_schema:
         type: "function"
         function:
           name: "code_interpreter"
           description: "Execute code in a secure cloud sandbox."
           parameters:
             type: "object"
             properties:
               code:
                 type: "string"
                 description: "The code to execute."
               language:
                 type: "string"
                 description: "Programming language (default: python)."
             required: ["code"]

Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------+--------------------------------------------------------------+------------------+
| Parameter               | Description                                                  | Default          |
+=========================+==============================================================+==================+
| ``image_id``            | Sandbox image to use                                         | ``code_latest``  |
+-------------------------+--------------------------------------------------------------+------------------+
| ``default_language``    | Default programming language for code execution              | ``python``       |
+-------------------------+--------------------------------------------------------------+------------------+
| ``api_key``             | AgentBay API key (can also be set via ``AGENTBAY_API_KEY``)  | None             |
+-------------------------+--------------------------------------------------------------+------------------+

How It Works
------------

The end-to-end RL pipeline with AgentBay:

1. **Prompt**: Model receives a math problem
2. **Generation**: Model generates reasoning and a ``code_interpreter`` tool call with Python code
3. **Execution**: AgentBay creates an isolated cloud sandbox, executes the code, returns the output
4. **Reward**: The model's final answer is scored against the ground truth
5. **Update**: GRPO policy gradient updates the model based on rewards

Each rollout trajectory gets its own isolated sandbox session, which is automatically
created and destroyed during the rollout.

Implementation
--------------

The tool implementation is at ``verl/tools/agentbay_tool.py``. It extends ``BaseTool``
and provides:

- ``create()``: Creates a new AgentBay cloud sandbox session
- ``execute()``: Runs code in the sandbox and returns the output
- ``release()``: Destroys the sandbox session

The SDK import is lazy-loaded, so ``wuying-agentbay-sdk`` is only required when
AgentBay is actually used in the tool configuration.
