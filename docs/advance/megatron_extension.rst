Add models with the Megatron-LM backend
=========================================

Last updated: 04/25/2025.

Model
-----------


If use latest verl, we have direct support of ``GPTModel`` for Megatron backend. 
You can use the similar way of using Megatron to pretrain custom models. 
We list the steps here:

1. Find `model_initializer.py <https://github.com/verl-project/verl/blob/main/verl/models/mcore/model_initializer.py>`_
2. If your model is configurable by ``TransformerLayerSpec`` , you can
   directly use ``GPTModel``. Otherwise, Please implement a new
   ``ModelLayerSpec`` and ``ModelLayer`` here.
3. Use the right ``LayerSpec`` , ``TransformerConfig`` and ``HuggingfaceConfig`` 
   as arguments to initialize the GPTModel.
4. Return the model at last.
