import json
from abc import abstractmethod, ABC
from collections import OrderedDict
from logging import Logger
from typing import List
from tqdm import tqdm
from transformers import BertTokenizer
import numpy as np
import logging

from spert import util
from spert.entities import Dataset, EntityType, RelationType, Entity, Relation, Document
from spert.opt import spacy

MAX_LENGTH = 512


class BaseInputReader(ABC):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, logger: Logger = None, **kwargs):


        types = json.load(open(types_path), object_pairs_hook=OrderedDict)  # entity + relation types


        self._entity_types = OrderedDict()
        self._idx2entity_type = OrderedDict()



        self._relation_types = OrderedDict()
        self._idx2relation_type = OrderedDict()

        self._sent_types = OrderedDict()
        self._idx2sent_label = OrderedDict()


        # entities
        # add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # specified entity types
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i + 1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i + 1] = entity_type


        # # subtypes
        # # add 'None' subtype
        # none_subtype = EntityType('None', 0, 'None', 'No Subtype')
        # self._subtypes['None'] = none_subtype
        # self._idx2subtype[0] = none_subtype
        #
        # # specified entity types
        # for i, (key, v) in enumerate(types['subtypes'].items()):
        #     subtype = EntityType(key, i + 1, v['short'], v['verbose'])
        #     self._subtypes[key] = subtype
        #     self._idx2subtype[i + 1] = subtype


        self._subtypes = OrderedDict()
        self._idx2subtype = OrderedDict()

        # define None for subtype
        none_subtype = EntityType('None', 0, 'None', 'No Subtype')

        # iterate over subtype layers
        for i, (layer_name, layer_def) in enumerate(types['subtypes'].items()):

            self._subtypes[layer_name] = OrderedDict()
            self._idx2subtype[layer_name] = OrderedDict()

            self._subtypes[layer_name]['None'] = none_subtype
            self._idx2subtype[layer_name][0] = none_subtype

            # iterate over labels in current layer subtype
            for i, (key, v) in enumerate(layer_def.items()):
                subtype = EntityType(key, i + 1, v['short'], v['verbose'])
                self._subtypes[layer_name][key] = subtype
                self._idx2subtype[layer_name][i + 1] = subtype



        # relations
        # add 'None' relation type
        none_relation_type = RelationType('None', 0, 'None', 'No Relation')
        self._relation_types['None'] = none_relation_type
        self._idx2relation_type[0] = none_relation_type

        # specified relation types
        for i, (key, v) in enumerate(types['relations'].items()):
            relation_type = RelationType(key, i + 1, v['short'], v['verbose'], v['symmetric'])
            self._relation_types[key] = relation_type
            self._idx2relation_type[i + 1] = relation_type

        if "sent_labels" not in types:
            sent_label_types = ['none']
        else:
            sent_label_types = types["sent_labels"]
        for i, key in enumerate(sent_label_types):
            self._sent_types[key] = i
            self._idx2sent_label[i] = key


        self._neg_entity_count = neg_entity_count
        self._neg_rel_count = neg_rel_count
        self._max_span_size = max_span_size

        self._datasets = dict()

        self._tokenizer = tokenizer
        self._logger = logger

        self._vocabulary_size = tokenizer.vocab_size

    @abstractmethod
    def read(self, dataset_path, dataset_label):
        pass

    def get_dataset(self, label) -> Dataset:
        return self._datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity

    def get_subtype(self, layer_name, idx) -> EntityType:
        subtype = self._idx2subtype[layer_name][idx]
        return subtype

    def get_relation_type(self, idx) -> RelationType:
        relation = self._idx2relation_type[idx]
        return relation

    def get_sent_label(self, idxs) -> dict:

        assert len(self._sent_types.keys()) == len(idxs)

        y = OrderedDict([(k, int(v)) for k, v in zip(self._sent_types.keys(), idxs)])
        return y

    # def get_word_piece_label(self, idxs) -> List[str]:
    #
    #     y = [self._idx2entity_type[i] for i in idxs]
    #
    #     return y

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def subtypes(self):
        return self._subtypes

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def relation_type_count(self):
        return len(self._relation_types)

    @property
    def entity_type_count(self):
        return len(self._entity_types)
    #
    # @property
    # def subtype_count(self):
    #     return len(self._subtypes)

    @property
    def subtype_count(self):
        return OrderedDict([(layer_name, len(layer_def.keys())) \
                        for layer_name, layer_def in self._subtypes.items()])


    @property
    def sent_type_count(self):
        return len(self._sent_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()


class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, logger: Logger = None):
        super().__init__(types_path, tokenizer, neg_entity_count, neg_rel_count, max_span_size, logger)

    def read(self, dataset_path, dataset_label):


        dataset = Dataset( \
                        label = dataset_label,
                        rel_types = self._relation_types,
                        entity_types = self._entity_types,
                        subtypes = self._subtypes,
                        sent_types = self._sent_types,
                        neg_entity_count = self._neg_entity_count,
                        neg_rel_count = self._neg_rel_count,
                        max_span_size = self._max_span_size)


        self._parse_dataset(dataset_path, dataset)
        self._datasets[dataset_label] = dataset
        return dataset


    def _parse_dataset(self, dataset_path, dataset):
        documents = json.load(open(dataset_path))
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    def _parse_document(self, doc, dataset) -> Document:
        jtokens = doc['tokens']
        jrelations = doc['relations']
        jentities = doc['entities']
        jsubtypes = doc['subtypes']

        if "sent_labels" in doc:
            jsent_labels = doc["sent_labels"]
        else:
            jsent_labels = None

        # parse tokens
        doc_tokens, doc_encoding = _parse_tokens(jtokens, dataset, self._tokenizer)

        

        assert len(doc_tokens) <= MAX_LENGTH
        assert len(doc_encoding) <= MAX_LENGTH

        # parse entity mentions
        entities, subtypes = self._parse_entities(jentities, jsubtypes, doc_tokens, dataset)

        # parse relations
        relations = self._parse_relations(jrelations, entities, dataset)

        


        sent_labels = self._parse_sent_labels(jsent_labels)

        word_piece_labels = self._parse_word_piece_labels(doc_encoding, entities)

        # create document
        document = dataset.create_document( \
                                    tokens = doc_tokens,
                                    entity_mentions = entities,
                                    subtype_mentions = subtypes,
                                    relations = relations,
                                    sent_labels = sent_labels,
                                    word_piece_labels = word_piece_labels,
                                    doc_encoding = doc_encoding)
        return document

    def _parse_word_piece_labels(self, doc_encoding, entities):

        entity_type_count = len(self.entity_types)
        sequence_length = len(doc_encoding)

        word_piece_labels = [('None', 0)]*sequence_length
        for entity in entities:
            start, end = entity.span
            label = entity.entity_type.identifier
            index = entity.entity_type.index

            for i in range(start, end):
                #if word_piece_labels[i] not in [0, label_index]:
                #    logging.warn(f"Overriding word piece label: {word_piece_labels[i]} --> {label_index}")
                word_piece_labels[i] = (label, index)

        return word_piece_labels



    def _parse_sent_labels(self, jsent_labels):
        if jsent_labels is None:
            y =  [0]*len(self._sent_types)
        else:
            y =  [jsent_labels[k] for k in self._sent_types]

        return y

    def _parse_entities(self, jentities, jsubtypes, doc_tokens, dataset) -> (List[Entity], List[Entity]):

        entities = []
        subtypes = []

        assert len(jentities) == len(jsubtypes)
        for entity_idx, (jentity, jsubtype) in enumerate(zip(jentities, jsubtypes)):

            # jentity is a dict
            #    keys: "type", "start", "end"

            # jsubtype is a dict
            #    keys: "type", "start", "end"

            # self._entity_types is a dict  mapping labels to class objects


            entity_type = self._entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']


            assert (start, end) == (jsubtype['start'], jsubtype['end'])

            subtype_dict = OrderedDict()
            for layer_name, layer_value in jsubtype['type'].items():
                if layer_value is None:
                    layer_value = "None"
                subtype_dict[layer_name] = self._subtypes[layer_name][layer_value]


            # create entity mention
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            # entity, subtype = dataset.create_entity(entity_type, subtype, tokens, phrase)
            entity, subtype = dataset.create_entity(entity_type, subtype_dict, tokens, phrase)
            entities.append(entity)
            subtypes.append(subtype)

        return (entities, subtypes)



    def _parse_relations(self, jrelations, entities, dataset) -> List[Relation]:
        relations = []

        for jrelation in jrelations:
            relation_type = self._relation_types[jrelation['type']]

            head_idx = jrelation['head']
            tail_idx = jrelation['tail']

            # create relation
            head = entities[head_idx]
            tail = entities[tail_idx]

            head_start, head_end = head.span
            tail_start, tail_end = tail.span


            # print(head_start, head_end, tail_start, tail_end)
            #print(head._tokens, head.span_start, head.span_end)


            reverse = int(tail.tokens[0].index) < int(head.tokens[0].index)

            # for symmetric relations: head occurs before tail in sentence
            if relation_type.symmetric and reverse:
                head, tail = util.swap(head, tail)

            relation = dataset.create_relation(relation_type, head_entity=head, tail_entity=tail, reverse=reverse)
            relations.append(relation)

        return relations


class JsonPredictionInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, spacy_model: str = None,
                 max_span_size: int = None, logger: Logger = None):
        super().__init__(types_path, tokenizer, max_span_size=max_span_size, logger=logger)
        self._spacy_model = spacy_model

        self._nlp = spacy.load(spacy_model) if spacy is not None and spacy_model is not None else None

    def read(self, dataset_path, dataset_label):
        dataset = Dataset(dataset_label, self._relation_types, self._entity_types, self._subtypes, self._neg_entity_count,
                          self._neg_rel_count, self._max_span_size)
        self._parse_dataset(dataset_path, dataset)
        self._datasets[dataset_label] = dataset
        return dataset

    def _parse_dataset(self, dataset_path, dataset):
        documents = json.load(open(dataset_path))
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    def _parse_document(self, document, dataset) -> Document:
        if type(document) == list:
            jtokens = document
        elif type(document) == dict:
            jtokens = document['tokens']
        else:
            jtokens = [t.text for t in self._nlp(document)]

        # parse tokens
        doc_tokens, doc_encoding = _parse_tokens(jtokens, dataset, self._tokenizer)

        # create document
        document = dataset.create_document(doc_tokens, [], [], doc_encoding)

        return document


def _parse_tokens(jtokens, dataset, tokenizer):
    doc_tokens = []

    # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
    doc_encoding = [tokenizer.convert_tokens_to_ids('[CLS]')]

    # parse tokens
    for i, token_phrase in enumerate(jtokens):

        token_encoding = tokenizer.encode(token_phrase, add_special_tokens=False)
        if not token_encoding:
            token_encoding = [tokenizer.convert_tokens_to_ids('[UNK]')]
        span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))

        token = dataset.create_token(i, span_start, span_end, token_phrase)

        # Avoid exceed max BERT length
        if len(doc_encoding) + len(token_encoding) < MAX_LENGTH - 1:

            doc_tokens.append(token)
            doc_encoding += token_encoding

    doc_encoding += [tokenizer.convert_tokens_to_ids('[SEP]')]

    return doc_tokens, doc_encoding
