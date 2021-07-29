from abc import ABC

import torch


from spert.models import NO_SUBTYPE, NO_CONCAT, CONCAT_LOGITS, CONCAT_PROBS, LABEL_BIAS


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class SpERTLoss(Loss):
    def __init__(self, rel_criterion, entity_criterion, sent_criterion, model, optimizer, scheduler, max_grad_norm, subtype_classification, include_sent_task):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        self._sent_criterion = sent_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm
        self._subtype_classification = subtype_classification
        self._include_sent_task = include_sent_task


    def compute(self, entity_logits, subtype_logits, rel_logits, sent_logits, entity_types, subtypes, rel_types, entity_sample_masks, rel_sample_masks, sent_labels):


        # entity loss
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        subtype_logits = subtype_logits.view(-1, subtype_logits.shape[-1])

        entity_types = entity_types.view(-1)
        subtypes = subtypes.view(-1)

        entity_sample_masks = entity_sample_masks.view(-1).float()

        entity_loss = self._entity_criterion(entity_logits, entity_types)

        entity_loss =  (entity_loss  * entity_sample_masks).sum() / entity_sample_masks.sum()


        # relation loss
        rel_sample_masks = rel_sample_masks.view(-1).float()
        rel_count = rel_sample_masks.sum()

        train_loss = entity_loss

        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])

            rel_loss = self._rel_criterion(rel_logits, rel_types)
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

            # joint loss
            train_loss += rel_loss

        if self._subtype_classification != NO_SUBTYPE:

            subtype_loss = self._entity_criterion(subtype_logits, subtypes)
            subtype_loss = (subtype_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

            # corner case: no positive/negative relation samples
            train_loss += subtype_loss

        if self._include_sent_task:
            sent_loss = self._sent_criterion(sent_logits, sent_labels).mean()

            train_loss += sent_loss


        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()
