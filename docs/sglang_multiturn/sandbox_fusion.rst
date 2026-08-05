===============================
Sandbox Fusion Tool Integration
===============================

Last updated: 05/09/2026.

.. note::

   The in-tree ``verl.tools.sandbox_fusion_tools.SandboxFusionTool`` reference
   implementation discussed in this guide has been removed. The integration
   pattern below remains accurate for users implementing their own
   :class:`verl.tools.base_tool.BaseTool` subclass; for the previous
   implementation see git history at commit ``f5e21df6`` and earlier.

Motivations
===========

- As users of verl, we want to allow the model to call certain tools during Actor rollout, incorporating the results into the training process.
- A colleague from ByteDance proposed a paper aimed at enhancing model capability through code execution tools.
- We aim to support tool-calling capabilities of inference engines using `sandbox-fusion` as the code execution system, providing the community with a reimplementation of `retools`.

Reward Compute with Sandbox Fusion + FaaS Integration
=====================================================

- In current datasets and tasks, similar work already exists (e.g., Prime), which uses local processes as runners to execute model-generated code for reward computation.
- On this basis, #1429 has advanced the design by integrating FaaS as the runner for reward computation.

Goals
=====

- Adapt to the `sglang` tool-calling protocol and define tools for sandbox fusion.
- Integrate with the `async-rollout` process, ensuring sandbox fusion tools follow asyncIO conventions.
- Design and implement a basic rate limiter to prevent issues such as 429 errors.

Non-Goals
=========

- Training effectiveness is out of scope.
- Observability metrics are not considered.
- Distributed failover and component fault tolerance are not addressed.

Design Details
==============

Tool Schema Definition
----------------------

- Currently, only code execution is considered, requiring a `code` field in the JSON from the model.
- Only Python code is supported for now, so no `language` parameter is defined.

.. code-block:: python

   OpenAIFunctionToolSchema(
       type="function",
       function=OpenAIFunctionSchema(
           name="code_interpreter",
           description="A tool for executing code.",
           parameters=OpenAIFunctionParametersSchema(
               type="object",
               properties={
                   "code": OpenAIFunctionPropertySchema(
                       type="string",
                       description="The code to execute.",
                       enum=None,
                   )
               },
               required=["code"],
           ),
           strict=False,
       )
   )

Configuration Parameters
--------------------------

+----------------------------+--------------------------------------------------------------+
| Parameter Name             | Description                                                  |
+============================+==============================================================+
| `num_workers`              | Number of worker threads/processes per DP to request runner. |
+----------------------------+--------------------------------------------------------------+
| `rate_limit`               | Global limit of concurrent code executions. Default: 10      |
+----------------------------+--------------------------------------------------------------+
| `default_timeout`          | Timeout (in seconds) for each code execution. Default: 30    |
+----------------------------+--------------------------------------------------------------+
| `default_language`         | Default programming language. Default: "python"              |
+----------------------------+--------------------------------------------------------------+
| `enable_global_rate_limit` | Whether to enable global rate limiting. Default: True        |
+----------------------------+--------------------------------------------------------------+
| `sandbox_fusion_url`       | URL for the veFaas sandbox execution service                 |
+----------------------------+--------------------------------------------------------------+

Rate Limiting Design
-----------------------

Objective:

- Limit the number of inflight requests using a token bucket model.

- Ensure ordered submission to code runners to avoid starvation due to backoff.

Design Highlights:

- Use Ray Global Actor as a singleton distributed counter at cluster level.
  
- Semaphore used for counting, with `acquire` and `release` in separate thread pools to preserve order.
  
- Use Ray’s cloud-pickle to serialize functions for decoupled `ExecutionWorker`.

.. code-block:: python

   @ray.remote(concurrency_groups={"acquire": 1,"release": 10})
   class TokenBucketWorker:
       def __init__(self, rate_limit: int):
           self.rate_limit = rate_limit
           self.current_count = 0
           self._semaphore = threading.Semaphore(rate_limit)

       @ray.method(concurrency_group="acquire")
       def acquire(self):
           self._semaphore.acquire()
           self.current_count += 1

       @ray.method(concurrency_group="release")
       def release(self):
           self._semaphore.release()
           self.current_count -= 1

       def get_current_count(self):
           return self.current_count

   class ExecutionWorker:
       def __init__(self, enable_global_rate_limit=True, rate_limit=10):
           self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

       def _init_rate_limit(self, rate_limit):
           return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

       def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
           with ExitStack() as stack:
               stack.callback(self.rate_limit_worker.release.remote)
               ray.get(self.rate_limit_worker.acquire.remote())
               try:
                   return fn(*fn_args, **fn_kwargs)
               except Exception as e:
                   logger.warning(f"Error when executing code: {e}")

   def init_execution_pool(num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode=PoolMode.ThreadMode):
       if mode == PoolMode.ThreadMode:
           return ray.remote(ExecutionWorker).options(max_concurrency=num_workers).remote(
               enable_global_rate_limit=enable_global_rate_limit,
               rate_limit=rate_limit
           )
       else:
           raise NotImplementedError("Process mode is not implemented yet")

Tool Implementation
-------------------

- Use `instance_id` to identify requests across multiple dialogue rounds.
  
- Use `execution_pool` to implement async invocation.
  
- Cleanup state after rollout completion.

.. code-block:: python

   class SandboxFusionTool(BaseTool):
       def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
           ...
           self.execution_pool = init_execution_pool(...)
           ...

       async def create(self, instance_id: Optional[str] = None, ...):
           ...

        async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
            code = parameters.get("code", "")
            timeout = parameters.get("timeout", self.default_timeout)
            language = parameters.get("language", self.default_language)
            if not isinstance(code, str):
                code = str(code)

            result = await self.execution_pool.execute.remote(self.execute_code,instance_id,code,timeout,language)
            self._instance_dict[instance_id]["reward"].append(result.strip())

            return result, result, {}

        def execute_code(self,instance_id,code,timeout=30,language="python"):
            result_status, metadata  = _process_single_case(0, None, None,self.sandbox_fusion_url, code, timeout, language)
            # we should always expect this since we don't have correct answer
            if metadata["run_status"] == "Finished":
                actual_output = metadata["stdout"] if metadata["stdout"] is not None else ""
                return actual_output
            else:
                return "no stdout here"

       async def calc_reward(self, instance_id: str, ...):
           ...

       async def release(self, instance_id: str, ...):
           ...

Test Plan
=========

Unit Tests
----------

- **test_tools_registration**: Test tool registration and initialization.
- **test_rollout_req_creation**: Validate that `AsyncRolloutReq` is built correctly.
- **test_over_size_case**: Ensure rollout terminates early when exceeding `max_seq_len`.
- **test_tool_call_basic_case**: Mock `sglang` output, validate tool call and result.
- **test_tool_call_batch_case**: Test batch processing of tool calls.
- **test_basic_multi_process_init**: Validate Ray global actor behaves as singleton.
- **TestSingleNodeRateLimiterCase**: Verify rate limiter works in single-node mode.
- **test_rotten_execution**: Ensure rate limiter recovers from function errors.
- **TestMultiNodeRateLimiterCase**: Verify behavior in multi-node environments.

e2e Tests
----------
we provide e2e test scripts in `tests/special_e2e` folder, named `tests/special_e2e/run_gsm8k_fsdp_sgl_multiturn_sf_tool.sh`

by setting 'trainer.rollout_data_dir' you can dump the rollout data to local disk. here is an sample taken from the rollout data:

.. code-block:: python

   {
     "input": "
     
     system\nYou are a math expert. You are given a question and you need to solve it step by step. Reasoning step by step before any tool call. You should use the `calc_gsm8k_reward` tool after step by step solving the question, before generate final answer at least once and refine your answer if necessary. Put your final answer in the format of `#### <answer>`.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"code_interpreter\", \"description\": \"A tool for executing code.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"code\": {\"type\": \"string\", \"description\": \"The code to execute.\", \"enum\": null}}, \"required\": [\"code\"]}, \"strict\": false}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n
     
     user\nJohn gets a bonus that's the same percentage every year.  Last year he made $100,000 and got a $10,000 bonus.  This year he makes $200,000.  How much will John make this year when adding both his total pay and bonus together? Let's think step by step and output the final answer after `####`.\n
     
     assistant\n",
     "output": "assistant\n<think>\nOkay, let's see. John gets a bonus that's the same percentage every year. Last year, he made $100,000 and got a $10,000 bonus. This year, he makes $200,000. The question is asking how much he'll make this year when adding both his total pay and bonus together.\n\nFirst, I need to figure out the percentage bonus he gets. Since the bonus is the same percentage each year, if I can find that percentage based on last year's details, I can apply it to this year's earnings to find this year's bonus. Then, I'll add that bonus to his total pay this year.\n\nLast year, he earned $100,000 and got a $10,000 bonus. To find the percentage, I can divide the bonus by his total earnings. So, 10,000 divided by 100,000. Let me calculate that.\n\n10,000 / 100,000 equals 0.1, which is 10% as a percentage. So, his bonus is 10% of his pay each year.\n\nThis year, he makes $200,000. So, the bonus this year would be 10% of 200,000. Let me compute that using the code interpreter to ensure accuracy.\n
     
     <tool_call>\n{\"name\": \"code_interpreter\", \"arguments\": {\"code\": \"total_pay_this_year = 200000\\nbonus_percentage = 10 / 100\\nbonus_this_year = total_pay_this_year * bonus_percentage\\ntotal_income_this_year = total_pay_this_year + bonus_this_year\\nprint(total_income_this_year)\", \"executes\": \"True\"}}\n</tool_call>\n
     
     tool\n220000.0\n\n
     
     assistant\n<think>\nSo the code shows that John's total income this year would be $220,000. This is consistent with the 10% bonus on last year's pay of $100,000, which was $10,000. Therefore, this year's bonus of $20,000 added to his total pay of $200,000 results in a total income of $220,000.\n\nI need to verify that there are no other factors at play. The problem states it's a constant percentage bonus each year, so there's no mention of changing percentages or other variables. Therefore, applying a straightforward percentage of 10% on this year's partial salary amount would result in $20,000. Adding that to his total pay gives $220,000. The calculation is correct and aligns with the question's parameters.\n</think>\n\n#### 220000.0",
     "score": 0,
     "step": 1
   }

here is the readable format version:

.. code-block:: python

   [system]
   
   You are a math expert. You are given a question and you need to solve it step by step. Reasoning step by step before any tool call. You should use the `calc_gsm8k_reward` tool after step by step solving the question, before generate final answer at least once and refine your answer if necessary. Put your final answer in the format of `#### <answer>`.
   
   # Tools
   
   You may call one or more functions to assist with the user query.
   
   You are provided with function signatures within <tools></tools> XML tags:
   <tools>
   {"type": "function", "function": {"name": "code_interpreter", "description": "A tool for executing code.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The code to execute.", "enum": null}}, "required": ["code"]}, "strict": false}}
   </tools>
   
   For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
   <tool_call>
   {"name": <function-name>, "arguments": <args-json-object>}
   </tool_call>
   
   [user]
   
   John gets a bonus that's the same percentage every year.  Last year he made $100,000 and got a $10,000 bonus.  This year he makes $200,000.  How much will John make this year when adding both his total pay and bonus together? Let's think step by step and output the final answer after `####`.
   
   [assistant]
   
   <think>
   Okay, let's see. John gets a bonus that's the same percentage every year. Last year, he made $100,000 and got a $10,000 bonus. This year, he makes $200,000. The question is asking how much he'll make this year when adding both his total pay and bonus together.
   
   First, I need to figure out the percentage bonus he gets. Since the bonus is the same percentage each year, if I can find that percentage based on last year's details, I can apply it to this year's earnings to find this year's bonus. Then, I'll add that bonus to his total pay this year.
   
   Last year, he earned $100,000 and got a $10,000 bonus. To find the percentage, I can divide the bonus by his total earnings. So, 10,000 divided by 100,000. Let me calculate that.
   
   10,000 / 100,000 equals 0.1, which is 10% as a percentage. So, his bonus is 10% of his pay each year.
   
   This year, he makes $200,000. So, the bonus this year would be 10% of 200,000. Let me compute that using the code interpreter to ensure accuracy.
   <tool_call>
   {"name": "code_interpreter", "arguments": {"code": "total_pay_this_year = 200000\nbonus_percentage = 10 / 100\nbonus_this_year = total_pay_this_year * bonus_percentage\ntotal_income_this_year = total_pay_this_year + bonus_this_year\nprint(total_income_this_year)", "executes": "True"}}
   </tool_call>
   
   [tool]
   
   220000.0
   
   [assistant]
   
   <think>
   So the code shows that John's total income this year would be $220,000. This is consistent with the 10% bonus on last year's pay of $100,000, which was $10,000. Therefore, this year's bonus of $20,000 added to his total pay of $200,000 results in a total income of $220,000.
   
   I need to verify that there are no other factors at play. The problem states it's a constant percentage bonus each year, so there's no mention of changing percentages or other variables. Therefore, applying a straightforward percentage of 10% on this year's partial salary amount would result in $20,000. Adding that to his total pay gives $220,000. The calculation is correct and aligns with the question's parameters.
   </think>
   
   #### 220000.0


You can also use the `RolloutViewer` TUI tool to view the dumped rollout data:


.. code-block:: bash

    python scripts/rollout_viewer.py ${trainer.rollout_data_dir}


.. image:: https://github.com/user-attachments/assets/e34e5157-2880-4a21-afb2-73885d0dfb11
   :alt: RolloutViewer screenshot