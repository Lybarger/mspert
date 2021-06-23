

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
HEAD = "head"
TAIL = "tail"

EXACT = "exact"
PARTIAL = "partial"
OVERLAP = "overlap"

NT = 'NT'
NP = 'NP'
TP = 'TP'

METRIC = 'metric'
COUNT = "count"


def PRF(df):

    df["P"] = df[TP]/(df[NP].astype(float))
    df["R"] = df[TP]/(df[NT].astype(float))
    df["F1"] = 2*df["P"]*df["R"]/(df["P"] + df["R"])

    return df

def augment_dict_keys(d, x):

    # get current keys
    keys = list(d.keys())

    # iterate over original keys
    for k_original in keys:

        # create new key
        k_new = k_original
        if isinstance(k_new, str):
            k_new = [k_new]
        k_new = tuple(list(k_new) + [x])

        # update dictionary
        d[k_new] = d.pop(k_original)

    return d

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


def merge_entities_relations(entities, relations):
    """
    Merge entities with relations
    """


    merged_relations = []
    for relation in relations:

        head_index = relation[HEAD]
        tail_index = relation[TAIL]

        head = entities[head_index]
        tail = entities[tail_index]

        d = {}
        d[TYPE] = relation[TYPE]
        d[HEAD] = head
        d[TAIL] = tail

        merged_relations.append(d)


    return merged_relations


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

def get_relation_counts(relations, \
                        head_scoring = EXACT,
                        tail_scoring = EXACT,
                        include_subtype = False):
    """
    Get histogram of entity labels
    """

    assert head_scoring in [EXACT, OVERLAP]
    assert tail_scoring in [EXACT, OVERLAP, PARTIAL]

    counter = Counter()
    for relation in relations:

        head = relation[HEAD]
        tail = relation[TAIL]

        # key for counter
        k = (relation[TYPE], head[TYPE], tail[TYPE])
        if include_subtype:
            k = tuple(list(k) + [head[SUBTYPE], tail[SUBTYPE]])

        # count spans
        if tail_scoring in [EXACT, OVERLAP]:
            counter[k] += 1

        # count tokens
        elif tail_scoring in [PARTIAL]:
            counter[k] += tail[END] - tail[START]

        else:
            raise ValueError(f"invalid tail scoring: {tail_scoring}")

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


def compare_entities(gold, pred, entity_scoring=EXACT, include_subtype=False):

    # assess label match
    type_match = gold[TYPE] == pred[TYPE]
    subtype_match = (not include_subtype) or (gold[SUBTYPE] == pred[SUBTYPE])
    type_match = type_match and subtype_match

    y = 0

    if type_match:
        g1 = gold[START]
        g2 = gold[END]

        p1 = pred[START]
        p2 = pred[END]

        indices_match = (g1, g2) == (p1, p2)
        indices_overlap = get_overlap_count(g1, g2, p1, p2)

        # exact match
        # count spans
        if (entity_scoring == EXACT) and indices_match:
            y = 1

        # partial match
        # count tokens
        elif (entity_scoring == PARTIAL) and indices_overlap:
            y = indices_overlap

        # any overlap match
        # count spans
        elif (entity_scoring == OVERLAP) and indices_overlap:
            y = 1

    return y





def compare_relations(gold, pred, \
                    head_scoring = EXACT,
                    tail_scoring = EXACT,
                    include_subtype = False):

    assert head_scoring in [EXACT, OVERLAP]
    assert tail_scoring in [EXACT, OVERLAP, PARTIAL]

    role_match = gold[TYPE] == pred[TYPE]

    head_match = compare_entities(gold[HEAD], pred[HEAD], \
                                    entity_scoring = head_scoring,
                                    include_subtype = include_subtype)

    tail_match = compare_entities(gold[TAIL], pred[TAIL], \
                                    entity_scoring = tail_scoring,
                                    include_subtype = include_subtype)

    if role_match and head_match and tail_match:
        return tail_match
    else:
        return 0




def get_entity_matches(gold, pred, entity_scoring=EXACT, include_subtype=False):

    assert entity_scoring in [EXACT, PARTIAL, OVERLAP]

    counter = Counter()

    for g in gold:

        for p in pred:

            v = compare_entities(g, p, \
                                entity_scoring = entity_scoring,
                                include_subtype = include_subtype)

            if v:
                k = (g[TYPE], g[SUBTYPE]) if include_subtype else g[TYPE]
                counter[k] += v

    return counter

def get_relation_matches(gold, pred, \
                                head_scoring = EXACT,
                                tail_scoring = EXACT,
                                include_subtype = False):

    assert head_scoring in [EXACT, OVERLAP]
    assert tail_scoring in [EXACT, OVERLAP, PARTIAL]

    counter = Counter()

    for g in gold:

        for p in pred:

            v = compare_relations(g, p, \
                                head_scoring = head_scoring,
                                tail_scoring = tail_scoring,
                                include_subtype = include_subtype)

            if v:
                # key for counter
                k = (g[TYPE], g[HEAD][TYPE], g[TAIL][TYPE])
                if include_subtype:
                    k = tuple(list(k) + [g[HEAD][SUBTYPE], g[TAIL][SUBTYPE]])

                counter[k] += v

    return counter


def score_entities(gold, pred, entity_scoring=EXACT, include_subtype=False):


    nt = get_entity_counts(gold, \
                                entity_scoring = entity_scoring,
                                include_subtype = include_subtype)

    np = get_entity_counts(pred, \
                                entity_scoring = entity_scoring,
                                include_subtype = include_subtype)

    tp = get_entity_matches(gold, pred, \
                                entity_scoring = entity_scoring,
                                include_subtype = include_subtype)

    counter = Counter()
    for name, d in [(NT, nt), (NP, np), (TP, tp)]:
        d = augment_dict_keys(d, name)
        counter.update(d)

    return counter


def score_relations(gold, pred, \
                    head_scoring = EXACT,
                    tail_scoring = EXACT,
                    include_subtype = False):


    nt = get_relation_counts(gold, \
                            head_scoring = head_scoring,
                            tail_scoring = tail_scoring,
                            include_subtype = include_subtype)

    np = get_relation_counts(pred, \
                            head_scoring = head_scoring,
                            tail_scoring = tail_scoring,
                            include_subtype = include_subtype)

    tp = get_relation_matches(gold, pred, \
                                    head_scoring = head_scoring,
                                    tail_scoring = tail_scoring,
                                    include_subtype = include_subtype)
    counter = Counter()
    for name, d in [(NT, nt), (NP, np), (TP, tp)]:
        d = augment_dict_keys(d, name)
        counter.update(d)

    return counter


def score_sent(gold, pred, \
                entity_scoring = EXACT,
                head_scoring = EXACT,
                tail_scoring = EXACT):


    gold_tokens = gold[TOKENS]
    pred_tokens =  pred[TOKENS]

    gold_entities = gold[ENTITIES]
    pred_entities =  pred[ENTITIES]

    gold_subtypes = gold[SUBTYPES]
    pred_subtypes =  pred[SUBTYPES]

    gold_relations = gold[RELATIONS]
    pred_relations = pred[RELATIONS]

    assert len(gold_tokens) == len(pred_tokens)
    assert gold_tokens == pred_tokens

    # merge sub types with entities
    gold_entities = merge_entity_subtypes(gold_entities, gold_subtypes)
    pred_entities = merge_entity_subtypes(pred_entities, pred_subtypes)

    entity_counts = score_entities(gold_entities, pred_entities, \
                        entity_scoring = entity_scoring,
                        include_subtype = False)

    subtype_counts = score_entities(gold_entities, pred_entities, \
                        entity_scoring = entity_scoring,
                        include_subtype = True)


    gold_relations = merge_entities_relations(gold_entities, gold_relations)
    pred_relations = merge_entities_relations(pred_entities, pred_relations)

    relation_counts1 = score_relations(gold_relations, pred_relations, \
                        head_scoring = head_scoring,
                        tail_scoring = tail_scoring,
                        include_subtype = False)

    relation_counts2 = score_relations(gold_relations, pred_relations, \
                        head_scoring = head_scoring,
                        tail_scoring = tail_scoring,
                        include_subtype = True)


    return (entity_counts, subtype_counts, relation_counts1, relation_counts2)

def get_entity_df(counts, include_subtype=False):

    cols = [TYPE]
    if include_subtype:
        cols = cols + [SUBTYPE]

    counts = [list(k) + [v] for k, v in counts.items()]
    df = pd.DataFrame(counts, columns= cols + [METRIC, COUNT])
    df = pd.pivot_table(df, values=COUNT, index=cols, columns=METRIC)
    df = df.fillna(0).astype(int)
    df = df.reset_index()
    df = df.sort_values(cols)
    df = df[cols + [NT, NP, TP]]
    df = PRF(df)
    df = df.fillna(0)

    return df

def get_relation_df(counts, include_subtype=False):

    cols = [TYPE, "ENTITY_TYPE_A", "ENTITY_TYPE_B"]
    if include_subtype:
        cols = cols + ["SUBTYPE_A", "SUBTYPE_B"]


    counts = [list(k) + [v] for k, v in counts.items()]
    df = pd.DataFrame(counts, columns= cols + [METRIC, COUNT])
    df = pd.pivot_table(df, values=COUNT, index=cols, columns=METRIC)
    df = df.fillna(0).astype(int)
    df = df.reset_index()
    df = df.sort_values(cols)
    df = df[cols + [NT, NP, TP]]
    df = PRF(df)
    df = df.fillna(0)

    return df



def score(gold, pred, \
                    entity_scoring = EXACT,
                    head_scoring = EXACT,
                    tail_scoring = EXACT):

    assert len(gold) == len(pred)

    entity_counts = Counter()
    subtype_counts = Counter()
    relation_counts1 = Counter()
    relation_counts2 = Counter()
    for gold_sent, pred_sent in zip(gold, pred):
        ec, sc, rc1, rc2 = score_sent(gold_sent, pred_sent, \
                                            entity_scoring = entity_scoring,
                                            head_scoring = head_scoring,
                                            tail_scoring = tail_scoring)
        entity_counts += ec
        subtype_counts += sc
        relation_counts1 += rc1
        relation_counts2 += rc2

    df_entity = get_entity_df(entity_counts, include_subtype=False)
    df_subtype = get_entity_df(subtype_counts, include_subtype=True)
    df_relation1 = get_relation_df(relation_counts1, include_subtype=False)
    df_relation2 = get_relation_df(relation_counts2, include_subtype=True)

    df_dict = OrderedDict()
    df_dict["entity"] = df_entity
    df_dict["subtype"] = df_subtype
    df_dict["relation"] = df_relation1
    df_dict["relation_subtype"] = df_relation2

    return df_dict



def score_files(gold_file, pred_file, \
                    entity_scoring = EXACT,
                    head_scoring = EXACT,
                    tail_scoring = EXACT):

    with open(gold_file, "r") as f:
        gold = json.load(f)

    with open(pred_file, "r") as f:
        pred = json.load(f)

    df_dict = score(gold, pred, \
                    entity_scoring = EXACT,
                    head_scoring = EXACT,
                    tail_scoring = EXACT)

    return df_dict


gold_file = "/home/lybarger/incidentalomas/analyses/step020_spert_datasets/radiology/dev_data.json"
pred_file = "/home/lybarger/incidentalomas/analyses/step102_anatomy_extraction/train/unknown/log/anatomy/2021-06-22_12:11:55.779989/predictions_valid_epoch_5.json"


score_files(gold_file, pred_file, \
                    entity_scoring = EXACT,
                    head_scoring = EXACT,
                    tail_scoring = EXACT)
