

import json
import pandas as pd
import os
import numpy as np
from collections import Counter, OrderedDict

TYPE = 'type'
SUBTYPE = 'subtype'
START = 'start'
END = 'end'
TOKENS = 'tokens'
ENTITIES = 'entities'
SUBTYPES = 'subtypes'
RELATIONS = 'relations'


EXACT = "exact"
PARTIAL = "partial"
OVERLAP = "overlap"


def entity_to_tuple(entity):
    return (entity[TYPE], entity[START], entity[END])

def merge_entity_subtypes(entities, subtypes):
    """
    Merge subtypes with entities
    """

    merged_entities = []

    # iterate over entities
    for entity in entities:

        # create new merged entity dictionary
        merged_entity = {}
        merged_entity.update(entity)
        merged_entity[SUBTYPE] = None

        # iterate over subtypes
        for subtype in subtypes:

            # update subif indices match
            if (entity[START], entity[END]) == (subtype[START], subtype[END]):
                merged_entity[SUBTYPE] = subtype[TYPE]
                break

        merged_entities.append(merged_entity)

    assert len(entities) == len(merged_entities)

    return merged_entities


def get_entity_counts(entities, entity_scoring=EXACT, include_subtype=False):
    """
    Get histogram of entity labels
    """

    counter = Counter()
    for entity in entities:

        # key for counter
        if include_subtype:
            k = (entity[TYPE], entity[SUBTYPE])
        else:
            k = entity[TYPE]

        # count spans
        if entity_scoring in [EXACT, OVERLAP]:
            counter[k] += 1

        # count tokens
        elif entity_scoring in [PARTIAL]:
            counter[k] += entity[END] - entity[START]

        else:
            raise ValueError(f"invalid entities scoring: {entity_scoring}")

    return counter

def get_overlap(i1, i2, j1, j2):
    """
    Get overlap between spans
    """

    A = set(range(i1, i2))
    B = set(range(j1, j2))
    overlap = A.intersection(B)
    overlap = sorted(list(overlap))

    return overlap

def get_overlap_count(i1, i2, j1, j2):
    """
    Determine if any overlap
    """
    overlap = get_overlap(i1, i2, j1, j2)
    return len(overlap)

def has_overlap(i1, i2, j1, j2):
    """
    Determine if any overlap
    """
    overlap = get_overlap(i1, i2, j1, j2)
    return len(overlap) > 0


def get_entity_matches(gold, pred, entity_scoring=EXACT, include_subtype=False):

    assert entity_scoring in [EXACT, PARTIAL, OVERLAP]

    counter = Counter()

    for g in gold:

        for p in pred:

            # assess label match
            type_match = g[TYPE] == p[TYPE]
            subtype_match = (not include_subtype) or (g[SUBTYPE] == p[SUBTYPE])
            type_match = type_match and subtype_match

            if type_match:

                # key for counter
                if include_subtype:
                    k = (g[TYPE], g[SUBTYPE])
                else:
                    k = g[TYPE]

                indices_match = (g[START], g[END]) == (p[START], p[END])
                indices_overlap = get_overlap_count(g[START], g[END], p[START], p[END])

                # exact match
                # count spans
                if (entity_scoring == EXACT) and indices_match:
                    counter[k] += 1

                # partial match
                # count tokens
                elif (entity_scoring == PARTIAL) and indices_overlap:
                    counter[k] += indices_overlap

                # any overlap match
                # count spans
                elif (entity_scoring == OVERLAP) and indices_overlap:
                    counter[k] += 1

    return counter



    return counter



def score_entities(gold, pred, entity_scoring=EXACT, include_subtype=False):


    NT = get_entity_counts(gold, \
                                entity_scoring = entity_scoring,
                                include_subtype = include_subtype)

    NP = get_entity_counts(pred, \
                                entity_scoring = entity_scoring,
                                include_subtype = include_subtype)

    TP = get_entity_matches(gold, pred, \
                                entity_scoring = entity_scoring,
                                include_subtype = include_subtype)

    return (NT, NP, TP)




def score_sent(gold, pred, entity_scoring=EXACT):


    gold_tokens = gold[TOKENS]
    pred_tokens =  pred[TOKENS]

    gold_entities = gold[ENTITIES]
    pred_entities =  pred[ENTITIES]

    gold_subtypes = gold[SUBTYPES]
    pred_subtypes =  pred[SUBTYPES]

    assert len(gold_tokens) == len(pred_tokens)
    assert gold_tokens == pred_tokens

    # merge sub types with entities
    gold_entities = merge_entity_subtypes(gold_entities, gold_subtypes)
    pred_entities = merge_entity_subtypes(pred_entities, pred_subtypes)

    score_entities(gold_entities, pred_entities, \
                        entity_scoring = entity_scoring,
                        include_subtype = False)


def score(gold_file, pred_file, entity_scoring=EXACT):

    with open(gold_file, "r") as f:
        gold = json.load(f)


    with open(pred_file, "r") as f:
        pred = json.load(f)


    assert len(gold) == len(pred)

    for gold_sent, pred_sent in zip(gold, pred):
        score_sent(gold_sent, pred_sent, \
                        entity_scoring = entity_scoring)


gold_file = "/home/lybarger/incidentalomas/analyses/step020_spert_datasets/radiology/dev_data.json"
pred_file = "/home/lybarger/incidentalomas/analyses/step102_anatomy_extraction/train/unknown/log/anatomy/2021-06-22_12:11:55.779989/predictions_valid_epoch_5.json"


score(gold_file, pred_file)
