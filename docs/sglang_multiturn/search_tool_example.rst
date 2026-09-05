=======================
Search Tool Integration
=======================

Last updated: 09/04/2026.

.. note::

   The in-tree ``verl.tools.search_tool.SearchTool`` reference implementation
   and the end-to-end recipe under ``examples/sglang_multiturn/`` (config,
   local retrieval server, and training scripts) have been removed from the
   tree. This page keeps a short conceptual overview of the former
   integration pattern; it is **not** a runnable quickstart. To rebuild a
   similar tool, subclass :class:`verl.tools.base_tool.BaseTool`.

Introduction
------------
Multi-Turn RL can call a search/retrieval tool during Actor rollout so the
model can use retrieval results for training. A typical setup uses a local
dense retriever or another local retrieval engine behind an HTTP service.

Removed recipe (formerly Quick Reproduction)
--------------------------------------------

The former Quick Reproduction steps cloned and ran paths under
``examples/sglang_multiturn/search_r1_like/`` (for example
``local_dense_retriever/download.py``,
``local_dense_retriever/retrieval_server.py``, and
``run_qwen2_5_3b_search_multiturn_fsdp.sh``). That directory tree is no
longer in the repository, so those commands are omitted here rather than
left as 404 paths.

Dataset preprocessing for a Search-R1-like format still exists at
``examples/data_preprocess/preprocess_search_r1_dataset.py`` if you are
building your own recipe.

Custom Search Configuration (conceptual)
----------------------------------------

To enable multi-turn reasoning with a tool-capable rollout backend, set
fields such as:

.. code:: yaml

   actor_rollout_ref:
     rollout:
       name: "sglang"
       multi_turn:
         enable: True

Provide your own tool YAML (the former
``examples/sglang_multiturn/config/tool_config/search_tool_config.yaml`` is
gone). Point ``class_name`` at a user-supplied ``BaseTool`` subclass, for
example:

.. code:: yaml

   tools:
     - class_name: my_package.tools.MySearchTool  # user-provided BaseTool subclass
       config:
         retrieval_service_url: http://127.0.0.1:8000/retrieve
         num_workers: 120
         rate_limit: 120
         timeout: 30

A common retriever input/output contract looks like:

.. code:: python

   Input format:
   {
     "queries": ["What is Python?", "Tell me about neural networks."],
     "topk": 3,
     "return_scores": true
   }

   Output format (when return_scores=True, similarity scores are returned):
   {
       "result": [
           [   # Results for each query
               {
                   "document": doc, "score": score
               },
               # ... more documents
           ],
           # ... results for other queries
       ]
   }

Notes
-----

1. The removed end-to-end recipe previously took on the order of ~27 hours
   for a full run, with a large validation set (~51 k) that could take
   ~6000 s per validation (hence ``val_before_train=False`` by default in
   that script).
