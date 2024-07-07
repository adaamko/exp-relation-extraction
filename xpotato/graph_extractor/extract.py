import json
import os
from collections import defaultdict

import networkx as nx
import pandas as pd
import penman as pn
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from tuw_nlp.graph.utils import GraphFormulaPatternMatcher

from xpotato.dataset.utils import amr_pn_to_graph, default_pn_to_graph, ud_to_graph

class GraphExtractor:
    def __init__(self, cache_dir=None, cache_fn=None, lang=None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        if cache_fn is None:
            cache_fn = os.path.join(cache_dir, f"{lang}_nlp_cache.json")
        self.cache_dir = cache_dir
        self.cache_fn = cache_fn
        self.lang = lang
        self.matcher = None

        self.ud_parser = None
        self.fl_parser = None
        self.amr_parser = None
        self.ucca_parser = None
        self.sdp_parser = None
        self.drs_parser = None

    def init_resources(self, graph_type):
        if graph_type == "ud":
            if self.ud_parser == None:
                from tuw_nlp.grammar.text_to_ud import TextToUD

                self.ud_parser = TextToUD(
                    lang=self.lang, nlp_cache=self.cache_fn, cache_dir=self.cache_dir
                )

        elif graph_type == "fourlang":
            if self.fl_parser == None:
                from tuw_nlp.grammar.text_to_4lang import TextTo4lang

                self.fl_parser = TextTo4lang(
                    lang=self.lang, nlp_cache=self.cache_fn, cache_dir=self.cache_dir
                )

        elif graph_type == "amr":
            if self.amr_parser == None:
                if self.lang != "en":
                    raise ValueError(
                        f"Currently only english AMR is supported: {self.lang}"
                    )
                from tuw_nlp.grammar.text_to_amr import TextToAMR

                self.amr_parser = TextToAMR()

        elif graph_type == "ucca":
            if self.ucca_parser == None:
                if self.lang != "en":
                    raise ValueError(
                        f"Currently only english UCCA is supported: {self.lang}"
                    )
                from tuw_nlp.grammar.text_to_ucca import TextToUCCA

                self.ucca_parser = TextToUCCA()

        elif graph_type == "sdp":
            if self.sdp_parser == None:
                if self.lang != "en":
                    raise ValueError(
                        f"Currently only english SDP is supported: {self.lang}"
                    )
                from tuw_nlp.grammar.text_to_sdp import TextToSDP

                self.sdp_parser = TextToSDP(lang=self.lang)
        elif graph_type == "drs":
            if self.drs_parser == None:
                if self.lang != "en":
                    raise ValueError(
                        f"Currently only english DRS is supported: {self.lang}"
                    )
                from tuw_nlp.grammar.text_to_drs import TextToDRS

                self.drs_parser = TextToDRS(lang=self.lang)

        else:
            raise ValueError(f"Currently not supported: {graph_type}")

    def parse_iterable(self, iterable, graph_type="fourlang", lang=None):
        if lang:
            self.lang = lang
        self.init_resources(graph_type)
        if graph_type == "fourlang":
            with self.fl_parser as tfl:
                for sen in tqdm(iterable):
                    fl_graphs = list(tfl(sen, ssplit=False))
                    g = fl_graphs[0]
                    for n in fl_graphs[1:]:
                        raise ValueError(f"sentence should not be split up: {sen}!")
                    yield g

        elif graph_type == "ud":
            for sen in tqdm(iterable):
                ud_graphs = list(self.ud_parser(sen, ssplit=False))
                g = ud_graphs[0]
                for n in ud_graphs[1:]:
                    raise ValueError(f"sentence should not be split up: {sen}!")
                yield g

        elif graph_type == "amr":
            for sen in tqdm(iterable):
                g = self.amr_parser(sen)

                yield list(g)[0]

        elif graph_type == "ucca":
            for sen in tqdm(iterable):
                g = self.ucca_parser(sen)

                yield list(g)[0]

        elif graph_type == "sdp":
            for sen in tqdm(iterable):
                g = self.sdp_parser(sen)

                yield list(g)[0]

        elif graph_type == "drs":
            for sen in tqdm(iterable):
                g = self.drs_parser(sen)

                yield list(g)[0]
        else:
            raise ValueError(f"Currrently not supported: {graph_type}")


class FeatureEvaluator:
    def __init__(self, graph_format="ud", case_sensitive=False):
        self.graph_format = graph_format
        self.case_sensitive = case_sensitive

    # ADAM: Very important to assign IDs to features from 0 because that's how
    # the mapping will work!!
    def annotate(self, graph, features):
        feature_to_marked_nodes = {}

        for i, feature in enumerate(features):
            assert (
                len(feature) == 4
            ), f"Feature must be a 4-tuple for OpenIE, not {feature}"

            positive_features = feature[0]
            negative_features = feature[1]

            for positive in positive_features:
                p = pn.decode(positive)
                first = p.triples[0][0]
                assert first == "u_0", f"The IDs must start from 0, not {first}"

            for negative in negative_features:
                p = pn.decode(negative)
                first = p.triples[0][0]
                assert first == "u_0", f"The IDs must start from 0, not {first}"

            feature_to_marked_nodes[i] = feature[3]
            features[i] = feature[:3]

        matcher = GraphFormulaPatternMatcher(
            features, converter=default_pn_to_graph, case_sensitive=self.case_sensitive
        )
        feats = matcher.match(graph, return_subgraphs=True)

        for key, i, subgraphs in feats:
            triplet = {"relation": key}
            marked_nodes = feature_to_marked_nodes[i]
            for j, node in enumerate(marked_nodes):
                subgraph = subgraphs[j]

                node_to_node = {}
                for id, graph_node in subgraph.nodes(data=True):
                    mapping = graph_node["mapping"]
                    node_to_node[mapping] = (
                        graph_node["entity"]
                        if "entity" in graph_node
                        else graph_node["name"]
                    )

                for k, v in node.items():
                    triplet[k] = node_to_node[v]

                yield triplet

    def annotate_dataframe(self, dataset, features):
        graphs = dataset.graph.tolist()

        triplets = []
        for graph in graphs:
            relations = self.annotate(graph, features)
            triplets.append(list(relations))
        d = {
            "Sentence": dataset.text.tolist(),
            "Triplets": triplets,
        }

        return pd.DataFrame(d)

    def match_features(
        self,
        dataset,
        features,
        multi=False,
        return_subgraphs=False,
        graph_matcher=GraphFormulaPatternMatcher,
        allow_multi_graph=False,
    ):
        graphs = dataset.graph.tolist()

        matches = []
        predicted = []
        matched_graphs = []

        matcher = graph_matcher(
            features, converter=default_pn_to_graph, case_sensitive=self.case_sensitive
        )

        for i, g in tqdm(enumerate(graphs)):
            feats = matcher.match(g, return_subgraphs=True)
            if multi:
                self.match_multi(
                    feats,
                    features,
                    matches,
                    predicted,
                    matched_graphs,
                    allow_multi_graph=allow_multi_graph,
                )
            else:
                self.match_not_multi(
                    feats, features, matches, predicted, matched_graphs
                )

        d = {
            "Sentence": dataset.text.tolist(),
            "Predicted label": predicted,
            "Matched rule": matches,
        }
        if return_subgraphs:
            d["Matched subgraph"] = matched_graphs

        df = pd.DataFrame(d)
        return df

    def match_multi(
        self,
        feats,
        features,
        matches,
        predicted,
        matched_graphs,
        allow_multi_graph=False,
    ):
        keys = []
        matched_rules = []
        matched_subgraphs = []
        for key, feature, graphs in feats:
            if key not in keys or allow_multi_graph:
                matched_rules.append(features[feature])
                matched_subgraphs.append(graphs)
                keys.append(key)
        if not keys:
            matches.append("")
            predicted.append("")
            matched_graphs.append("")
        else:
            matches.append(matched_rules)
            predicted.append(keys)
            matched_graphs.append(matched_subgraphs)

    def match_not_multi(self, feats, features, matches, predicted, matched_graphs):
        for key, feature, graphs in feats:
            matches.append(features[feature])
            predicted.append(key)
            matched_graphs.append(graphs)
            break
        else:
            matches.append("")
            matched_graphs.append("")
            predicted.append("")

    def one_versus_rest(self, df, entity):
        mapper = {entity: 1}

        one_versus_rest_df = df.copy()
        one_versus_rest_df["one_versus_rest"] = [
            mapper[item] if item in mapper else 0 for item in df.label
        ]

        return one_versus_rest_df

    def rank_features(self, cl, features, orig_data, false_negatives):
        # TODO: currently disabled
        # if false_negatives:
        #     subset_data = orig_data.iloc[false_negatives]
        # else:
        #   subset_data = orig_data
        subset_data = orig_data
        df, accuracy = self.evaluate_feature(cl, features, subset_data)

        features_stat = []

        for i, feature in enumerate(features):
            features_stat.append(
                (
                    feature,
                    df.iloc[i].Precision,
                    df.iloc[i].Recall,
                    df.iloc[i].Fscore,
                    df.iloc[i].Support,
                    len(df.iloc[i].True_positive_sens),
                    len(df.iloc[i].False_positive_sens),
                )
            )

        def rank(feature):
            return len(df.iloc[features.index(feature[0])].True_positive_graphs)

        return sorted(features_stat, key=rank, reverse=True)

    def train_feature(self, cl, feature, data, graph_format="ud"):
        graph_matcher = GraphFormulaPatternMatcher(
            [[[feature], [], []]],
            default_pn_to_graph,
            case_sensitive=self.case_sensitive,
        )
        feat_patt = graph_matcher.patts[0][0]
        if isinstance(feat_patt[0], tuple):
            if len(feat_patt[0][1]) == 2:
                patt1, patt2 = feat_patt[0][1]
            else:
                _, patt1, patt2 = feat_patt[0][1]
        else:
            patt1, patt2 = feat_patt[0], None

        graphs = data.graph.tolist()
        labels = self.one_versus_rest(data, cl).one_versus_rest.tolist()
        path = "trained_features.tsv"
        trained_features = []
        with open(path, "w+") as f:
            for i, g in enumerate(graphs):
                matches = [
                    (i, subgraph)
                    for (key, i, subgraph) in graph_matcher.match(
                        g, return_subgraphs=True
                    )
                ]
                for patt_index, match in matches:
                    for graph in match:
                        nodes = []
                        for node_index, node in graph.nodes(data=True):
                            if not nodes:
                                node_name = node["name"]
                                if (
                                    node["mapping"] in patt1.nodes
                                    and patt1.nodes[node["mapping"]]["name"] == ".*"
                                ):
                                    nodes.append(node_name)
                                if (
                                    patt2 is not None
                                    and node["mapping"] in patt2.nodes
                                    and patt2.nodes[node["mapping"]]["name"] == ".*"
                                ):
                                    nodes.append(node_name)
                        if not nodes:
                            g2_to_g1 = {v: u for (u, v, _) in graph.edges(data=True)}
                            for u, v, attrs in patt1.edges(data=True):
                                if attrs["color"] == ".*":
                                    edge = g.get_edge_data(g2_to_g1[u], g2_to_g1[v])[
                                        "color"
                                    ]
                                    nodes.append(edge)
                            for u, v, attrs in patt2.edges(data=True):
                                if attrs["color"] == ".*":
                                    edge = g.get_edge_data(g2_to_g1[u], g2_to_g1[v])[
                                        "color"
                                    ]
                                    nodes.append(edge)
                        nodes_str = ",".join(nodes)
                        label = labels[i]
                        sentence = data.iloc[i].text
                        f.write(f"{feature}\t{nodes_str}\t{sentence}\t{label}\n")
                        trained_features.append(
                            (feature, nodes_str, sentence, str(label))
                        )

        return self.cluster_feature(trained_features)

    def cluster_feature(self, trained_features):
        graphs = {}
        if os.path.isfile("longman_zero_paths_one_exp"):
            with open("longman_zero_paths_one_exp.json") as f:
                graphs = json.load(f)

        words = {}
        for fields in trained_features:
            words[fields[1] + "_" + fields[3]] = int(fields[3])
            feature = fields[0]
        graph = nx.MultiDiGraph()

        for word in words:
            if words[word] == 1:
                color = "green"
            else:
                color = "red"
            graph.add_node(word, color=color)
            word_clean = word.split("_")[0]
            if word_clean in graphs:
                hypernyms = graphs[word_clean]
                for hypernym in hypernyms:
                    hypernym_words = hypernyms[hypernym]
                    for w in hypernym_words:
                        if hypernym == "1":
                            graph.add_edge(word, w, color=hypernym)

        selected_words = self.select_words(trained_features)

        word_features = []

        if selected_words:
            word_features.append(feature.replace(".*", "|".join(selected_words), 1))
        else:
            word_features.append(feature)

        return word_features

    def select_words(self, trained_features):
        features = []
        labels = []

        for fields in trained_features:
            features.append(fields[1])
            labels.append(int(fields[3]))
        words_to_measures = {
            word: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for word in set(features)
        }
        for word in words_to_measures:
            for i, label in enumerate(labels):
                if label and features[i] == word:
                    words_to_measures[word]["TP"] += 1
                if label and features[i] != word:
                    words_to_measures[word]["FN"] += 1
                if not label and features[i] == word:
                    words_to_measures[word]["FP"] += 1
                if not label and features[i] != word:
                    words_to_measures[word]["TN"] += 1

        for word in words_to_measures:
            TP = words_to_measures[word]["TP"]
            FP = words_to_measures[word]["FP"]
            TN = words_to_measures[word]["TN"]
            FN = words_to_measures[word]["FN"]

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

            words_to_measures[word]["precision"] = precision
            words_to_measures[word]["recall"] = recall

        selected_words = set()

        for word in words_to_measures:
            if words_to_measures[word]["precision"] > 0.9 and (
                words_to_measures[word]["TP"] > 1
                or words_to_measures[word]["recall"] > 0.01
            ):
                selected_words.add(word)

        return selected_words

    def evaluate_feature(
        self,
        cl,
        features,
        data,
        graph_format="ud",
        graph_matcher=GraphFormulaPatternMatcher,
    ):
        measure_features = []
        graphs = data.graph.tolist()
        labels = self.one_versus_rest(data, cl).one_versus_rest.tolist()

        whole_predicted = []
        matched = defaultdict(list)

        # We want to view false negative examples for all rules, not rule specific
        false_neg_g = []
        false_neg_s = []
        false_neg_indices = []
        matcher = graph_matcher(
            features, converter=default_pn_to_graph, case_sensitive=self.case_sensitive
        )
        for i, g in enumerate(graphs):
            feats = matcher.match(g)
            label = 0
            for key, feature in feats:
                matched[i].append(features[feature][0])
                label = 1
            whole_predicted.append(label)

            if label == 0 and labels[i] == 1:
                false_neg_g.append(g)
                sen = data.iloc[i].text
                lab = data.iloc[i].label
                false_neg_s.append((sen, lab))
                false_neg_indices.append(i)

        accuracy = []
        for pcf in precision_recall_fscore_support(
            labels, whole_predicted, average=None
        ):
            if len(pcf) > 1:
                accuracy.append(pcf[1])
            else:
                accuracy.append(0)

        for feat in features:
            measure = [feat[0]]
            false_pos_g = []
            false_pos_s = []
            false_pos_indices = []
            true_pos_g = []
            true_pos_s = []
            true_pos_indices = []
            predicted = []
            for i, g in enumerate(graphs):
                feats = matched[i]
                label = 1 if feat[0] in feats else 0
                if label == 1 and labels[i] == 0:
                    false_pos_g.append(g)
                    sen = data.iloc[i].text
                    lab = data.iloc[i].label
                    false_pos_s.append((sen, lab))
                    false_pos_indices.append(i)
                if label == 1 and labels[i] == 1:
                    true_pos_g.append(g)
                    sen = data.iloc[i].text
                    lab = data.iloc[i].label
                    true_pos_s.append((sen, lab))
                    true_pos_indices.append(i)
                predicted.append(label)
            for pcf in precision_recall_fscore_support(labels, predicted, average=None):
                if len(pcf) > 1:
                    measure.append(pcf[1])
                else:
                    measure.append(0)
            measure.append(false_pos_g)
            measure.append(false_pos_s)
            measure.append(false_pos_indices)
            measure.append(true_pos_g)
            measure.append(true_pos_s)
            measure.append(true_pos_indices)
            measure.append(false_neg_g)
            measure.append(false_neg_s)
            measure.append(false_neg_indices)
            measure.append(predicted)
            measure_features.append(measure)

        df = pd.DataFrame(
            measure_features,
            columns=[
                "Feature",
                "Precision",
                "Recall",
                "Fscore",
                "Support",
                "False_positive_graphs",
                "False_positive_sens",
                "False_positive_indices",
                "True_positive_graphs",
                "True_positive_sens",
                "True_positive_indices",
                "False_negative_graphs",
                "False_negative_sens",
                "False_negative_indices",
                "Predicted",
            ],
        )

        return df, accuracy
