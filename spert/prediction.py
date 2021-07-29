import json
from typing import Tuple

import torch

from spert import util
from spert.input_reader import BaseInputReader


def convert_predictions(batch_entity_clf: torch.tensor, batch_subtype_clf: torch.tensor, batch_rel_clf: torch.tensor,
                        batch_rels: torch.tensor, batch_sent_clf: torch.tensor, batch: dict, rel_filter_threshold: float,
                        input_reader: BaseInputReader, no_overlapping: bool = False):
    # get maximum activation (index of predicted entity type)
    batch_entity_types = batch_entity_clf.argmax(dim=-1)
    # apply entity sample mask
    batch_entity_types *= batch['entity_sample_masks'].long()

    # get maximum activation (index of predicted entity type)
    batch_subtypes = batch_subtype_clf.argmax(dim=-1)
    # apply entity sample mask
    batch_subtypes *= batch['entity_sample_masks'].long()


    # apply threshold to relations
    batch_rel_clf[batch_rel_clf < rel_filter_threshold] = 0

    batch_sent_labels = torch.round(torch.sigmoid(batch_sent_clf))


    batch_pred_entities = []
    batch_pred_subtypes = []
    batch_pred_relations = []
    batch_pred_sent_labels = []

    for i in range(batch_rel_clf.shape[0]):
        # get model predictions for sample
        entity_types = batch_entity_types[i]
        subtypes = batch_subtypes[i]

        entity_spans = batch['entity_spans'][i]

        entity_clf = batch_entity_clf[i]
        subtype_clf = batch_subtype_clf[i]

        rel_clf = batch_rel_clf[i]
        rels = batch_rels[i]

        sent_labels = batch_sent_labels[i]

        # convert predicted entities
        sample_pred_entities = _convert_pred_entities(entity_types, entity_spans,
                                                      entity_clf, input_reader,
                                                      is_entity=True)

        sample_pred_subtypes = _convert_pred_entities(subtypes, entity_spans,
                                                      subtype_clf, input_reader,
                                                      is_entity=False)

        # convert predicted relations
        sample_pred_relations = _convert_pred_relations(rel_clf, rels,
                                                        entity_types, entity_spans, input_reader)

        if no_overlapping:
            sample_pred_subtypes, _ = remove_overlapping(sample_pred_subtypes,
                                                            sample_pred_relations)

            sample_pred_entities, sample_pred_relations = remove_overlapping(sample_pred_entities,
                                                                             sample_pred_relations)

        sample_pred_sent_labels = _convert_pred_sent_labels(sent_labels, input_reader)


        batch_pred_entities.append(sample_pred_entities)
        batch_pred_subtypes.append(sample_pred_subtypes)
        batch_pred_relations.append(sample_pred_relations)
        batch_pred_sent_labels.append((sample_pred_sent_labels))

    return batch_pred_entities, batch_pred_subtypes, batch_pred_relations, batch_pred_sent_labels


def _convert_pred_entities(entity_types: torch.tensor, entity_spans: torch.tensor,
                           entity_scores: torch.tensor, input_reader: BaseInputReader,
                           is_entity=True):
    # get entities that are not classified as 'None'
    valid_entity_indices = entity_types.nonzero().view(-1)
    pred_entity_types = entity_types[valid_entity_indices]
    pred_entity_spans = entity_spans[valid_entity_indices]
    pred_entity_scores = torch.gather(entity_scores[valid_entity_indices], 1,
                                      pred_entity_types.unsqueeze(1)).view(-1)

    # convert to tuples (start, end, type, score)
    converted_preds = []
    for i in range(pred_entity_types.shape[0]):
        label_idx = pred_entity_types[i].item()

        if is_entity:
            entity_type = input_reader.get_entity_type(label_idx)
        else:
            entity_type = input_reader.get_subtype(label_idx)

        start, end = pred_entity_spans[i].tolist()
        score = pred_entity_scores[i].item()

        converted_pred = (start, end, entity_type, score)
        converted_preds.append(converted_pred)

    return converted_preds


def _convert_pred_relations(rel_clf: torch.tensor, rels: torch.tensor,
                            entity_types: torch.tensor, entity_spans: torch.tensor, input_reader: BaseInputReader):
    rel_class_count = rel_clf.shape[1]
    rel_clf = rel_clf.view(-1)

    # get predicted relation labels and corresponding entity pairs
    rel_nonzero = rel_clf.nonzero().view(-1)
    pred_rel_scores = rel_clf[rel_nonzero]

    pred_rel_types = (rel_nonzero % rel_class_count) + 1  # model does not predict None class (+1)
    valid_rel_indices = rel_nonzero // rel_class_count
    valid_rels = rels[valid_rel_indices]

    # get masks of entities in relation
    pred_rel_entity_spans = entity_spans[valid_rels].long()

    # get predicted entity types
    pred_rel_entity_types = torch.zeros([valid_rels.shape[0], 2])
    if valid_rels.shape[0] != 0:
        pred_rel_entity_types = torch.stack([entity_types[valid_rels[j]] for j in range(valid_rels.shape[0])])

    # convert to tuples ((head start, head end, head type), (tail start, tail end, tail type), rel type, score))
    converted_rels = []
    check = set()

    for i in range(pred_rel_types.shape[0]):
        label_idx = pred_rel_types[i].item()
        pred_rel_type = input_reader.get_relation_type(label_idx)
        pred_head_type_idx, pred_tail_type_idx = pred_rel_entity_types[i][0].item(), pred_rel_entity_types[i][1].item()
        pred_head_type = input_reader.get_entity_type(pred_head_type_idx)
        pred_tail_type = input_reader.get_entity_type(pred_tail_type_idx)
        score = pred_rel_scores[i].item()

        spans = pred_rel_entity_spans[i]
        head_start, head_end = spans[0].tolist()
        tail_start, tail_end = spans[1].tolist()

        converted_rel = ((head_start, head_end, pred_head_type),
                         (tail_start, tail_end, pred_tail_type), pred_rel_type)
        converted_rel = _adjust_rel(converted_rel)

        if converted_rel not in check:
            check.add(converted_rel)
            converted_rels.append(tuple(list(converted_rel) + [score]))

    return converted_rels

def _convert_pred_sent_labels(sent_labels: torch.tensor, input_reader: BaseInputReader):

    sent_labels = sent_labels.tolist()
    sent_labels = input_reader.get_sent_label(sent_labels)
    return sent_labels



def remove_overlapping(entities, relations):
    non_overlapping_entities = []
    non_overlapping_relations = []

    for entity in entities:
        if not _is_overlapping(entity, entities):
            non_overlapping_entities.append(entity)

    for rel in relations:
        e1, e2 = rel[0], rel[1]
        if not _check_overlap(e1, e2):
            non_overlapping_relations.append(rel)

    return non_overlapping_entities, non_overlapping_relations


def _is_overlapping(e1, entities):
    for e2 in entities:
        if _check_overlap(e1, e2):
            return True

    return False


def _check_overlap(e1, e2):
    if e1 == e2 or e1[1] <= e2[0] or e2[1] <= e1[0]:
        return False
    else:
        return True


def _adjust_rel(rel: Tuple):
    adjusted_rel = rel
    if rel[-1].symmetric:
        head, tail = rel[:2]
        if tail[0] < head[0]:
            adjusted_rel = tail, head, rel[-1]

    return adjusted_rel


def store_predictions(documents, pred_entities, pred_subtypes, pred_relations, pred_sent_labels, store_path):
    predictions = []

    for i, doc in enumerate(documents):
        tokens = doc.tokens
        sample_pred_entities = pred_entities[i]
        sample_pred_subtypes = pred_subtypes[i]
        sample_pred_relations = pred_relations[i]
        sample_pred_sent_labels = pred_sent_labels[i]

        # convert entities
        converted_entities = []
        for j, entity in enumerate(sample_pred_entities):
            entity_span = entity[:2]
            span_tokens = util.get_span_tokens(tokens, entity_span)
            entity_type = entity[2].identifier
            converted_entity = dict(type=entity_type, start=span_tokens[0].index, end=span_tokens[-1].index + 1)
            converted_entities.append(converted_entity)
        converted_entities = sorted(converted_entities, key=lambda e: e['start'])

        converted_subtypes = []
        for j, subtype in enumerate(sample_pred_subtypes):
            subtype_span = subtype[:2]
            span_tokens = util.get_span_tokens(tokens, subtype_span)
            subtype_type = subtype[2].identifier
            converted_subtype = dict(type=subtype_type, start=span_tokens[0].index, end=span_tokens[-1].index + 1)
            converted_subtypes.append(converted_subtype)
        converted_subtypes = sorted(converted_subtypes, key=lambda e: e['start'])

        # convert relations
        converted_relations = []
        for relation in sample_pred_relations:
            head, tail = relation[:2]
            head_span, head_type = head[:2], head[2].identifier
            tail_span, tail_type = tail[:2], tail[2].identifier
            head_span_tokens = util.get_span_tokens(tokens, head_span)
            tail_span_tokens = util.get_span_tokens(tokens, tail_span)
            relation_type = relation[2].identifier

            converted_head = dict(type=head_type, start=head_span_tokens[0].index,
                                  end=head_span_tokens[-1].index + 1)
            converted_tail = dict(type=tail_type, start=tail_span_tokens[0].index,
                                  end=tail_span_tokens[-1].index + 1)

            head_idx = converted_entities.index(converted_head)
            tail_idx = converted_entities.index(converted_tail)

            converted_relation = dict(type=relation_type, head=head_idx, tail=tail_idx)
            converted_relations.append(converted_relation)
        converted_relations = sorted(converted_relations, key=lambda r: r['head'])

        converted_sent_labels = sample_pred_sent_labels

        doc_predictions = dict( \
                                tokens = [t.phrase for t in tokens],
                                entities = converted_entities,
                                subtypes = converted_subtypes,
                                relations = converted_relations,
                                sent_labels = converted_sent_labels)
        predictions.append(doc_predictions)

    # store as json
    with open(store_path, 'w') as predictions_file:
        json.dump(predictions, predictions_file)

    return predictions
