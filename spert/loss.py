from abc import ABC

import torch
import logging

from spert.models import NO_SUBTYPE, NO_CONCAT, CONCAT_LOGITS, CONCAT_PROBS, LABEL_BIAS


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class SpERTLoss(Loss):
    def __init__(self, rel_criterion, entity_criterion, sent_criterion, word_piece_criterion, model, optimizer, scheduler, max_grad_norm, subtype_classification, include_sent_task, include_word_piece_task):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        self._sent_criterion = sent_criterion
        self._word_piece_criterion = word_piece_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm
        self._subtype_classification = subtype_classification
        self._include_sent_task = include_sent_task
        self._include_word_piece_task = include_word_piece_task


    def compute(self, entity_logits, rel_logits, entity_types, rel_types, entity_sample_masks, rel_sample_masks, \
            subtype_logits = None,
            subtype_labels = None,
            sent_logits = None,
            sent_labels = None,
            # word_piece_logits = None,
            # word_piece_labels = None,
            # word_piece_mask = None
            ):


        # aggregate loss across all objectives
        total_loss = []

        '''
        Entity loss
        '''
        # entity loss
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])

        entity_types = entity_types.view(-1)


        entity_sample_masks = entity_sample_masks.view(-1).float()

        entity_loss = self._entity_criterion(entity_logits, entity_types)
        entity_loss =  (entity_loss  * entity_sample_masks).sum() / entity_sample_masks.sum()

        total_loss.append(entity_loss)

        # relation loss
        rel_sample_masks = rel_sample_masks.view(-1).float()
        rel_count = rel_sample_masks.sum()



        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])

            rel_loss = self._rel_criterion(rel_logits, rel_types)
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

            # joint loss
            # total_loss += rel_loss
            total_loss.append(rel_loss)

        if self._subtype_classification != NO_SUBTYPE:

            # subtype_logits - a dictionary of logits,
            #       dictionary keys are subtype layers
            #       logit tensor has shape (batch_size, span_count, num_classes)
            # subtype_labels - tensor for all subtype layers
            #       logic tensor has shape (batch_size, span_count, num_layers)

            batch_size, span_count, num_layers = tuple(subtype_labels.shape)
            assert num_layers == len(subtype_logits.keys())

            subtype_loss = []
            for i, (layer_name, sub_log) in enumerate(subtype_logits.items()):

                # (batch_size, span_count)
                sub_lab = subtype_labels[:,:,i].squeeze()

                # (batch_size*span_count)
                sub_lab = sub_lab.view(-1)

                # (batch_size, span_count, num_classes)
                # sub_log

                # (batch_size * span_count, num_classes)
                sub_log = sub_log.view(-1, sub_log.shape[-1])

                # (batch_size * span_count)
                sub_loss = self._entity_criterion(sub_log, sub_lab)

                # scalar
                sub_loss = (sub_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

                # collect loss across layers
                subtype_loss.append(sub_loss)

            # calculate total subtype loss
            subtype_loss = torch.stack(subtype_loss)
            subtype_loss = subtype_loss.sum()

            # corner case: no positive/negative relation samples
            # total_loss += subtype_loss
            total_loss.append(subtype_loss)

        if self._include_sent_task:
            sent_loss = self._sent_criterion(sent_logits, sent_labels).mean()
            # total_loss += sent_loss
            total_loss.append(sent_loss)


        # sum loss across all objectives
        total_loss = torch.stack(total_loss)
        total_loss = total_loss.sum()

        # logging.warn(f"entity={entity_loss:.3f}, subtype={subtype_loss:.3f}, total={total_loss:.3f}")
        # if self._include_word_piece_task:
        #
        #     word_piece_logits = word_piece_logits.view(-1, word_piece_logits.shape[-1])
        #     word_piece_labels = word_piece_labels.view(-1)
        #     m = word_piece_mask.view(-1).bool()
        #
        #     word_piece_loss = self._word_piece_criterion(word_piece_logits[m], word_piece_labels[m]).mean()
        #     total_loss += word_piece_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return total_loss.item()
