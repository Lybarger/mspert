import os
import os
import warnings
from typing import List, Tuple, Dict
from collections import OrderedDict
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from transformers import BertTokenizer

from spert import prediction
from spert.entities import Document, Dataset, EntityType
from spert.input_reader import BaseInputReader
from spert.opt import jinja2

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class Evaluator:
    def __init__(self, dataset: Dataset, input_reader: BaseInputReader, text_encoder: BertTokenizer,
                 rel_filter_threshold: float, no_overlapping: bool,
                 predictions_path: str, examples_path: str, example_count: int):
        self._text_encoder = text_encoder
        self._input_reader = input_reader
        self._dataset = dataset
        self._rel_filter_threshold = rel_filter_threshold
        self._no_overlapping = no_overlapping

        self._predictions_path = predictions_path
        self._examples_path = examples_path

        self._example_count = example_count

        # relations
        self._gt_relations = []  # ground truth
        self._pred_relations = []  # prediction

        # entities
        self._gt_entities = []  # ground truth
        self._pred_entities = []  # prediction

        self._gt_subtypes = []  # ground truth
        self._pred_subtypes = []  # prediction

        self._gt_sent_labels = []  # ground truth
        self._pred_sent_labels = []  # prediction

        # self._gt_word_piece_labels = []  # ground truth
        # self._pred_word_piece_labels = []  # prediction

        self._pseudo_entity_type = EntityType('Entity', 1, 'Entity', 'Entity')  # for span only evaluation

        self._convert_gt(self._dataset.documents)




    def eval_batch(self, batch_entity_clf: torch.tensor, batch_subtype_clf: torch.tensor, batch_rel_clf: torch.tensor,
                   batch_rels: torch.tensor, batch_sent_clf: torch.tensor, batch: dict):
                   # batch_rels: torch.tensor, batch_sent_clf: torch.tensor, batch_word_piece_clf: torch.tensor, batch: dict):




        entities, subtypes, relations, sent_labels = prediction.convert_predictions( \
                                batch_entity_clf = batch_entity_clf,
                                batch_subtype_clf = batch_subtype_clf,
                                batch_rel_clf = batch_rel_clf,
                                batch_rels = batch_rels,
                                batch_sent_clf = batch_sent_clf,
                                # batch_word_piece_clf = batch_word_piece_clf,
                                batch = batch,
                                rel_filter_threshold = self._rel_filter_threshold,
                                input_reader = self._input_reader,
                                no_overlapping = self._no_overlapping)

        self._pred_entities.extend(entities)
        self._pred_subtypes.extend(subtypes)
        self._pred_relations.extend(relations)
        self._pred_sent_labels.extend(sent_labels)
        # self._pred_word_piece_labels.extend(word_piece_labels)

    def compute_scores(self):
        print("Evaluation")


        print("")
        print("--- Word piece labels ---")
        print("")
        #self._score_word_piece_labels(self._gt_word_piece_labels, self._pred_word_piece_labels, print_results=True)

        print("")
        print("--- Entities (named entity recognition (NER)) ---")
        print("An entity is considered correct if the entity type and span is predicted correctly")
        print("")

        gt, pred = self._convert_by_setting(self._gt_entities, self._pred_entities, include_entity_types=True)
        ner_eval = self._score(gt, pred, print_results=True)



        print("")
        print("--- Subtypes (named entity recognition (NER)) ---")
        print("An entity is considered correct if the entity type and span is predicted correctly")
        print("")



        layer_names = self._input_reader._subtypes.keys()

        gt_subtypes = OrderedDict([(n, []) for n in layer_names])
        for doc in self._gt_subtypes[0:20]:

            for n in layer_names:
                gt_subtypes[n].append([])

            for ent_dict in doc[0:10]:
                for n, (start, end, ent) in ent_dict.items():
                    gt_subtypes[n][-1].append((start, end, ent))



        pred_subtypes = OrderedDict([(n, []) for n in layer_names])
        for doc in self._pred_subtypes[0:20]:

            for n in layer_names:
                pred_subtypes[n].append([])

            for (start, end, ent_dict) in doc:
                for n, ent in ent_dict.items():
                    pred_subtypes[n][-1].append((start, end, ent))


        for layer_name in gt_subtypes:
            print("")
            print(f"Subtype layer = {layer_name}")
            print("")

            gt = gt_subtypes[layer_name]
            pred = pred_subtypes[layer_name]

            gt, pred = self._convert_by_setting(gt, pred, include_entity_types=True)
            ner_eval_st = self._score(gt, pred, print_results=True)

        print("")
        print("--- Relations ---")
        print("")
        print("Without named entity classification (NEC)")
        print("A relation is considered correct if the relation type and the spans of the two "
              "related entities are predicted correctly (entity type is not considered)")
        print("")
        gt, pred = self._convert_by_setting(self._gt_relations, self._pred_relations, include_entity_types=False)
        rel_eval = self._score(gt, pred, print_results=True)

        print("")
        print("With named entity classification (NEC)")
        print("A relation is considered correct if the relation type and the two "
              "related entities are predicted correctly (in span and entity type)")
        print("")
        gt, pred = self._convert_by_setting(self._gt_relations, self._pred_relations, include_entity_types=True)
        rel_nec_eval = self._score(gt, pred, print_results=True)

        return ner_eval, ner_eval_st, rel_eval, rel_nec_eval

    def store_predictions(self):
        predictions = prediction.store_predictions( \
                        documents = self._dataset.documents,
                        pred_entities = self._pred_entities,
                        pred_subtypes = self._pred_subtypes,
                        pred_relations = self._pred_relations,
                        pred_sent_labels = self._pred_sent_labels,
                        store_path = self._predictions_path)

        return predictions

    def store_examples(self):
        if jinja2 is None:
            warnings.warn("Examples cannot be stored since Jinja2 is not installed.")
            return

        entity_examples = []
        rel_examples = []
        rel_examples_nec = []

        for i, doc in enumerate(self._dataset.documents):
            # entities
            entity_example = self._convert_example(doc, self._gt_entities[i], self._pred_entities[i],
                                                   include_entity_types=True, to_html=self._entity_to_html)
            entity_examples.append(entity_example)

            # relations
            # without entity types
            rel_example = self._convert_example(doc, self._gt_relations[i], self._pred_relations[i],
                                                include_entity_types=False, to_html=self._rel_to_html)
            rel_examples.append(rel_example)

            # with entity types
            rel_example_nec = self._convert_example(doc, self._gt_relations[i], self._pred_relations[i],
                                                    include_entity_types=True, to_html=self._rel_to_html)
            rel_examples_nec.append(rel_example_nec)

        # entities
        self._store_examples(entity_examples[:self._example_count],
                             file_path=self._examples_path % 'entities',
                             template='entity_examples.html')

        self._store_examples(sorted(entity_examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % 'entities_sorted',
                             template='entity_examples.html')

        # relations
        # without entity types
        self._store_examples(rel_examples[:self._example_count],
                             file_path=self._examples_path % 'rel',
                             template='relation_examples.html')

        self._store_examples(sorted(rel_examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % 'rel_sorted',
                             template='relation_examples.html')

        # with entity types
        self._store_examples(rel_examples_nec[:self._example_count],
                             file_path=self._examples_path % 'rel_nec',
                             template='relation_examples.html')

        self._store_examples(sorted(rel_examples_nec[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % 'rel_nec_sorted',
                             template='relation_examples.html')

    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            gt_relations = doc.relations

            # list of Entity objects
            gt_entities = doc.entities

            # list of dict
            #   dict keys - layer_name
            #   dict vals - Entity object
            gt_subtypes = doc.subtypes

            assert len(gt_entities) == len(gt_subtypes)

            # convert ground truth relations and entities for precision/recall/f1 evaluation

            sample_gt_entities = [entity.as_tuple() for entity in gt_entities]


            # sample_gt_subtypes = [subtype.as_tuple() for subtype in gt_subtypes]
            # if len(gt_subtypes) > 0:
            #     print(gt_subtypes)
            #     z = sldfkj
            sample_gt_subtypes = []
            for S in gt_subtypes:
                S = OrderedDict([(layer_name, s.as_tuple()) for layer_name, s in S.items()])
                sample_gt_subtypes.append(S)

            sample_gt_relations = [rel.as_tuple() for rel in gt_relations]

            # if self._no_overlapping:
            if False:
                # for layer_name,
                # sample_gt_subtypes, _ = prediction.remove_overlapping(sample_gt_subtypes,
                #                                                                         sample_gt_relations)
                #
                # sample_gt_entities, sample_gt_relations = prediction.remove_overlapping(sample_gt_entities,
                #                                                                         sample_gt_relations)

                sample_gt_entities, sample_gt_relations, sample_gt_subtypes = \
                            prediction.remove_overlapping_new(sample_gt_entities,
                                                        sample_gt_relations,
                                                        sample_gt_subtypes)


            self._gt_entities.append(sample_gt_entities)
            self._gt_subtypes.append(sample_gt_subtypes)
            self._gt_relations.append(sample_gt_relations)
            self._gt_sent_labels.append(doc.sent_labels)
            #self._gt_word_piece_labels.append(doc.word_piece_labels)

    def _convert_by_setting(self, gt: List[List[Tuple]], pred: List[List[Tuple]],
                            include_entity_types: bool = True, include_score: bool = False):
        assert len(gt) == len(pred)

        # either include or remove entity types based on setting
        def convert(t):
            if not include_entity_types:
                # remove entity type and score for evaluation
                if type(t[0]) == int:  # entity
                    c = [t[0], t[1], self._pseudo_entity_type]
                else:  # relation
                    c = [(t[0][0], t[0][1], self._pseudo_entity_type),
                         (t[1][0], t[1][1], self._pseudo_entity_type), t[2]]
            else:
                c = list(t[:3])

            if include_score and len(t) > 3:
                # include prediction scores
                c.append(t[3])

            return tuple(c)

        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):
            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])

        return converted_gt, converted_pred

    def _score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False):
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []
        types = set()

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)

            for s in union:
                if s in sample_gt:
                    t = s[2]
                    gt_flat.append(t.index)
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(t.index)
                    types.add(t)
                else:
                    pred_flat.append(0)

        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)
        return metrics

    def _score_word_piece_labels(self, gt, pred, print_results=False):


        gt_flat = []
        pred_flat = []
        assert len(gt) == len(pred)
        for g, p in zip(gt, pred):
            assert len(g) == len(p)
            gt_flat.extend(g)
            pred_flat.extend(p)

        labels = sorted(list(set(gt_flat + pred_flat)))
        labels.remove('None')

        per_type = prfs(gt_flat, pred_flat, labels=labels, average=None, zero_division=0)
        micro =    prfs(gt_flat, pred_flat, labels=labels, average='micro', zero_division=0)[:-1]
        macro =    prfs(gt_flat, pred_flat, labels=labels, average='macro', zero_division=0)[:-1]
        total_support = sum(per_type[-1])
        #print(per_type)

        #print(micro)

        columns=["type", "precision", "recall", "f1", "support"]
        columns=["precision", "recall", "f1", "support"]
        df_per_type = pd.DataFrame(dict(zip(columns, per_type)))
        df_per_type.insert(0, 'type', labels)

        df_micro = pd.DataFrame([micro], columns=["precision", "recall", "f1"])

        if print_results:
            print(df_per_type)
            print(df_micro)

        return True




    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False):
        labels = [t.index for t in types]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None, zero_division=0)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro', zero_division=0)[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro', zero_division=0)[:-1]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

        return [m * 100 for m in micro + macro]

    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        results = [row_fmt % columns, '\n']

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            results.append(row_fmt % self._get_row(m, t.short_name))
            results.append('\n')

        results.append('\n')

        # micro
        results.append(row_fmt % self._get_row(micro, 'micro'))
        results.append('\n')

        # macro
        results.append(row_fmt % self._get_row(macro, 'macro'))

        results_str = ''.join(results)
        print(results_str)

    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)

    def _convert_example(self, doc: Document, gt: List[Tuple], pred: List[Tuple],
                         include_entity_types: bool, to_html):
        encoding = doc.encoding

        gt, pred = self._convert_by_setting([gt], [pred], include_entity_types=include_entity_types, include_score=True)
        gt, pred = gt[0], pred[0]

        # get micro precision/recall/f1 scores
        if gt or pred:
            pred_s = [p[:3] for p in pred]  # remove score
            precision, recall, f1 = self._score([gt], [pred_s])[:3]
        else:
            # corner case: no ground truth and no predictions
            precision, recall, f1 = [100] * 3

        scores = [p[-1] for p in pred]
        pred = [p[:-1] for p in pred]
        union = set(gt + pred)

        # true positives
        tp = []
        # false negatives
        fn = []
        # false positives
        fp = []

        for s in union:
            type_verbose = s[2].verbose_name

            if s in gt:
                if s in pred:
                    score = scores[pred.index(s)]
                    tp.append((to_html(s, encoding), type_verbose, score))
                else:
                    fn.append((to_html(s, encoding), type_verbose, -1))
            else:
                score = scores[pred.index(s)]
                fp.append((to_html(s, encoding), type_verbose, score))

        tp = sorted(tp, key=lambda p: p[-1], reverse=True)
        fp = sorted(fp, key=lambda p: p[-1], reverse=True)

        text = self._prettify(self._text_encoder.decode(encoding))
        return dict(text=text, tp=tp, fn=fn, fp=fp, precision=precision, recall=recall, f1=f1, length=len(doc.tokens))

    def _entity_to_html(self, entity: Tuple, encoding: List[int]):
        start, end = entity[:2]
        entity_type = entity[2].verbose_name

        tag_start = ' <span class="entity">'
        tag_start += '<span class="type">%s</span>' % entity_type

        ctx_before = self._text_encoder.decode(encoding[:start])
        e1 = self._text_encoder.decode(encoding[start:end])
        ctx_after = self._text_encoder.decode(encoding[end:])

        html = ctx_before + tag_start + e1 + '</span> ' + ctx_after
        html = self._prettify(html)

        return html

    def _rel_to_html(self, relation: Tuple, encoding: List[int]):
        head, tail = relation[:2]
        head_tag = ' <span class="head"><span class="type">%s</span>'
        tail_tag = ' <span class="tail"><span class="type">%s</span>'

        if head[0] < tail[0]:
            e1, e2 = head, tail
            e1_tag, e2_tag = head_tag % head[2].verbose_name, tail_tag % tail[2].verbose_name
        else:
            e1, e2 = tail, head
            e1_tag, e2_tag = tail_tag % tail[2].verbose_name, head_tag % head[2].verbose_name

        segments = [encoding[:e1[0]], encoding[e1[0]:e1[1]], encoding[e1[1]:e2[0]],
                    encoding[e2[0]:e2[1]], encoding[e2[1]:]]

        ctx_before = self._text_encoder.decode(segments[0])
        e1 = self._text_encoder.decode(segments[1])
        ctx_between = self._text_encoder.decode(segments[2])
        e2 = self._text_encoder.decode(segments[3])
        ctx_after = self._text_encoder.decode(segments[4])

        html = (ctx_before + e1_tag + e1 + '</span> '
                + ctx_between + e2_tag + e2 + '</span> ' + ctx_after)
        html = self._prettify(html)

        return html

    def _prettify(self, text: str):
        text = text.replace('_start_', '').replace('_classify_', '').replace('<unk>', '').replace('⁇', '')
        text = text.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
        return text

    def _store_examples(self, examples: List[Dict], file_path: str, template: str):
        template_path = os.path.join(SCRIPT_PATH, 'templates', template)

        # read template
        with open(os.path.join(SCRIPT_PATH, template_path)) as f:
            template = jinja2.Template(f.read())

        # write to disc
        template.stream(examples=examples).dump(file_path)
