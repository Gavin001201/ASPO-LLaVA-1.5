import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from .base_dpo_trainer import BaseDPOTrainer

class LlavaDPOTrainer(BaseDPOTrainer):
        
    def concatenated_forward(
        self, model, inputs, adaptive_weight=False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        images = inputs["images"]
        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_labels = inputs["chosen_labels"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        chosen_sentence_masks = inputs["chosen_sentence_masks"]
        chosen_sentence_scores = inputs["chosen_sentence_scores"]
        reject_input_ids = inputs["reject_input_ids"]
        reject_labels = inputs["reject_labels"]
        reject_attention_mask = inputs["reject_attention_mask"]
        reject_sentence_masks = inputs["reject_sentence_masks"]
        reject_sentence_scores = inputs["reject_sentence_scores"]
            
            
        # chosen_sentence_masks = torch.zeros_like(chosen_labels, dtype=torch.int)
        batch_size, sequence_length = chosen_labels.shape
        # # 遍历每个样本
        # for batch_idx in range(batch_size):
        #     current_label = 1
        #     for seq_idx in range(sequence_length):
        #         if (chosen_labels[batch_idx, seq_idx] == -100) or (chosen_labels[batch_idx, seq_idx] == 2):
        #             chosen_sentence_masks[batch_idx, seq_idx] = -100
        #         else:
        #             chosen_sentence_masks[batch_idx, seq_idx] = current_label
        #         if chosen_labels[batch_idx, seq_idx] == 29889:  # 如果遇到句号，则增加标签编号
        #             current_label += 1
                    
 
        # reject_sentence_masks = torch.zeros_like(reject_labels, dtype=torch.int)   
        # batch_size, sequence_length = reject_labels.shape
        # # 遍历每个样本
        # for batch_idx in range(batch_size):
        #     current_label = 1
        #     for seq_idx in range(sequence_length):
        #         if (reject_labels[batch_idx, seq_idx] == -100) or (reject_labels[batch_idx, seq_idx] == 2):
        #             reject_sentence_masks[batch_idx, seq_idx] = -100
        #         else:
        #             reject_sentence_masks[batch_idx, seq_idx] = current_label
        #         if reject_labels[batch_idx, seq_idx] == 29889:  # 如果遇到句号，则增加标签编号
        #             current_label += 1
            
            
        max_dim = max(chosen_input_ids.shape[1], reject_input_ids.shape[1])
        batch_input_ids = torch.zeros((chosen_input_ids.shape[0]*2, max_dim), dtype=chosen_input_ids.dtype, device=chosen_input_ids.device)
        batch_labels = torch.ones((chosen_input_ids.shape[0]*2, max_dim), dtype=chosen_labels.dtype, device=chosen_labels.device) * -100
        batch_attention_mask = torch.zeros((chosen_input_ids.shape[0]*2, max_dim), device=chosen_attention_mask.device).to(torch.bool)
        batch_input_ids[:chosen_input_ids.shape[0], :chosen_input_ids.shape[1]] = chosen_input_ids
        batch_input_ids[reject_input_ids.shape[0]:, :reject_input_ids.shape[1]] = reject_input_ids
        batch_labels[:chosen_labels.shape[0], :chosen_labels.shape[1]] = chosen_labels
        batch_labels[reject_labels.shape[0]:, :reject_labels.shape[1]] = reject_labels
        batch_attention_mask[:chosen_attention_mask.shape[0], :chosen_attention_mask.shape[1]] = chosen_attention_mask
        batch_attention_mask[reject_attention_mask.shape[0]:, :reject_attention_mask.shape[1]] = reject_attention_mask
        
        chosen_step_masks = torch.ones((batch_size, max_dim), dtype=chosen_labels.dtype, device=chosen_labels.device) * -100
        reject_step_masks = torch.ones((batch_size, max_dim), dtype=chosen_labels.dtype, device=chosen_labels.device) * -100
        chosen_step_masks[:, :chosen_sentence_masks.shape[1]] = chosen_sentence_masks
        reject_step_masks[:, :reject_sentence_masks.shape[1]] = reject_sentence_masks    
        
        image_palcehoder = torch.ones((batch_size, 575), dtype=torch.int, device=chosen_labels.device) * -100
        chosen_step_masks = torch.cat((image_palcehoder, chosen_step_masks), dim=1)
        reject_step_masks = torch.cat((image_palcehoder, reject_step_masks), dim=1)
        
        step_masks = torch.cat((chosen_step_masks, reject_step_masks), dim=0).float()
        
        sentence_scores = chosen_sentence_scores + reject_sentence_scores
        
        # prepare inputs
        (
            batch_input_ids,
            batch_position_ids,
            batch_attention_mask,
            batch_past_key_values,
            batch_inputs_embeds,
            batch_labels
        ) = self.model.prepare_inputs_labels_for_multimodal(
            input_ids=batch_input_ids,
            position_ids=None,
            attention_mask=batch_attention_mask,
            past_key_values=None,
            labels=batch_labels,
            images=torch.cat([images, images], dim=0),
        )
        
        # calculate logits
        all_logits = model.forward(
            inputs_embeds=batch_inputs_embeds,
            labels=None,
            attention_mask=batch_attention_mask,
        ).logits.to(torch.float32)
        cal_batch_logp = self._get_batch_logps
        all_logps = cal_batch_logp(
            all_logits,
            batch_labels,
            step_masks,
            sentence_scores,
            adaptive_weight=adaptive_weight,
            average_log_prob=False,
        )
        
        len_chosen = chosen_input_ids.shape[0]
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        
        # don't count image embeds logits
        loss_mask = batch_labels != -100
        logits = [all_logits[i][loss_mask[i]] for i in range(loss_mask.shape[0])]
        chosen_logits = logits[:len_chosen]
        rejected_logits = logits[len_chosen:]
        chosen_logits = [l.detach().cpu().mean() for l in chosen_logits]
        rejected_logits = [l.detach().cpu().mean() for l in rejected_logits]
        chosen_logits = sum(chosen_logits)/len_chosen
        rejected_logits = sum(rejected_logits)/len_chosen
        
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_metrics(
        self,
        inputs,
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}
        
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(self.model, inputs, adaptive_weight=True)
        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, inputs, adaptive_weight=True)

        policy_rejected_logps = policy_rejected_logps
        reference_rejected_logps = reference_rejected_logps
           
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"policy_{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"policy_{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"referece_{prefix}logps/rejected"] = reference_rejected_logps.detach().cpu().mean()
        metrics[f"referece_{prefix}logps/chosen"] = reference_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits

        return losses.mean(), metrics
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
            
        loss, metrics = self.get_batch_metrics(inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss
