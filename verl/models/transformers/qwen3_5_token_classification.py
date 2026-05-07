import torch
from torch import nn
from transformers import Qwen2Model, Qwen2PreTrainedModel, Qwen2Config
from transformers.modeling_outputs import TokenClassifierOutput

class Qwen3_5ForTokenClassification(Qwen2PreTrainedModel):
    config_class = Qwen2Config 

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # The Backbone
        self.model = Qwen2Model(config)
        # The Head
        self.dropout = nn.Dropout(getattr(config, "classifier_dropout", 0.1))
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if not return_dict:
            # We take our logits and add them to the rest of the outputs from the backbone
            output = (logits,) + outputs[2:]
            return output

        return TokenClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
