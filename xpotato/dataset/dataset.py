import json
import re
from re import I
from typing import Dict, List, Tuple
import spacy
import amrlib

import networkx as nx
import pandas as pd
from networkx.readwrite import json_graph
from tqdm import tqdm
from tuw_nlp.graph.utils import check_if_str_is_penman, graph_to_pn
from tuw_nlp.grammar.text_to_4lang import TextTo4lang
from tuw_nlp.grammar.text_to_ud import TextToUD
from tuw_nlp.graph.amr_graph import AMRGraph

from xpotato.dataset.sample import Sample
from xpotato.graph_extractor.extract import GraphExtractor
from xpotato.graph_extractor.graph import PotatoGraph


class Dataset:
    def __init__(
        self,
        examples: List[Tuple[str, str]] = None,
        label_vocab: Dict[str, int] = {},
        lang="en",
        path=None,
        binary=False,
        cache_dir=None,
        cache_fn=None,
    ) -> None:
        self.label_vocab = label_vocab
        if path:
            self._dataset = self.read_dataset(path=path, binary=binary)
        else:
            self._dataset = self.read_dataset(examples=examples)

        self.extractor = GraphExtractor(
            lang=lang, cache_dir=cache_dir, cache_fn=cache_fn
        )
        self.graphs = None

    @staticmethod
    def save_dataframe(df: pd.DataFrame, path: str) -> None:
        graphs = [
            json.dumps(json_graph.adjacency_data(g)) if type(g) == nx.DiGraph else g
            for g in df["graph"].tolist()
        ]
        df["graph"] = graphs
        df.to_csv(path, index=False, sep="\t")

    @staticmethod
    def generate_dataframe_ud_fl(df, sentence_colname="", label_id_colname="", sentence_id_colname=""):
        """
        Submit a dataframe and compute a response dataframe with entity-marked graphs
        Expects columns b1,e1 b2,e2 to provide begin/end character positions of the respective entities 1 and 2.
        :param df: The dataframe containing the input sentence and all necessary conversion information
        :param sentence_colname: which column is the sentence in
        :param label_id_colname: which column is the label in
        :param sentence_id_colname: which column is the unique sentence id in
        :return: a new dataframe with the above information, converted and entity-tagged UD and FL graphs as well as a
                 report for how many and which entity tokens could be mapped to their respective FL nodes
        """
        def is_allcaps(text):
            return not re.search(r'[a-z]', text)

        extractor = GraphExtractor(
            lang="en", cache_dir=None, cache_fn=None
        )

        ud_parser = TextToUD(
            lang=extractor.lang, nlp_cache=extractor.cache_fn, cache_dir=extractor.cache_dir
        )

        fl_parser = TextTo4lang(
            lang=extractor.lang, nlp_cache=extractor.cache_fn, cache_dir=extractor.cache_dir
        )

        rows = df.iterrows()
        sentences = [row[1][sentence_colname] for row in rows]
        rows = df.iterrows()
        label_ids = [row[1][label_id_colname] for row in rows]
        rows = df.iterrows()
        sentence_ids = [row[1][sentence_id_colname] for row in rows]

        ud_graph_list = []
        fl_graphs = []
        reports = []
        e1_found = []
        e2_found = []

        for i, sent in enumerate(sentences):

            # For each graph keep track of which entity tokens are linked to nodes, and which couldn't be found
            entity1_dict = {}
            entity2_dict = {}
            tokens_e1 = 0
            tokens_e1_found = 0
            tokens_e2 = 0
            tokens_e2_found = 0
            e1_not_found = []
            e2_not_found = []

            # Parse UD
            ud_graphs = list(ud_parser(sent))
            num_sents = len(ud_graphs)

            # If there are more than one sentences, take the last one. This seems to discard only "et al." introductions, which are irrelevant
            ud_graph = ud_graphs[num_sents - 1]
            fl_graph = list(fl_parser(sent))[num_sents - 1]

            # 4lang graph nodes have a different index but the same token IDs as the UD graph they are built from, so we need this mapping
            node_ids = {}
            for idx in fl_graph.G.nodes:
                t_id = fl_graph.G.nodes[idx]['token_id']
                node_ids[t_id] = idx

            # Go through the tokens and check for each token if it is part of a relationship entity. If yes, mark the node.
            for t in ud_graph.ud_graph.tokens:
                # The node is an entity node if the associated start and end character positions fall within
                # the entity bounds defined in the dataframe OR if one of them does and the word is ALLCAPS.
                # This second condition is necessary because the CrowdTruth dataset gives us inaccurate indexes.

                # Entity 1
                if (t.start_char >= df.iloc[i].b1 and t.end_char <= df.iloc[i].e1) \
                        or (df.iloc[i].b1 <= t.start_char <= df.iloc[i].e1 and is_allcaps(t.text)) \
                        or (df.iloc[i].b1 <= t.end_char <= df.iloc[i].e1 and is_allcaps(t.text)):
                    # Tag UD
                    id = t.id[0]
                    ud_graph.G.nodes[id]["entity"] = 1

                    # If the FL graph has a node with the same ID, we can tag it as well
                    tokens_e1 += 1
                    if id in node_ids.keys():
                        node_id = node_ids[id]
                        fl_graph.G.nodes[node_id]["entity"] = 1
                        tokens_e1_found += 1
                        entity1_dict[t.text] = fl_graph.G.nodes[node_id]
                    else:
                        e1_not_found.append(t.text)

                # Entity 2
                elif (t.start_char >= df.iloc[i].b2 and t.end_char <= df.iloc[i].e2) or (
                        t.start_char >= df.iloc[i].b2 and t.start_char <= df.iloc[i].e2 and is_allcaps(t.text)) or (
                        t.end_char >= df.iloc[i].b2 and t.end_char <= df.iloc[i].e2 and is_allcaps(t.text)):
                    # Tag UD
                    id = t.id[0]
                    ud_graph.G.nodes[id]["entity"] = 2

                    # If the FL graph has a node with the same ID, we can tag it as well
                    tokens_e2 += 1
                    if id in node_ids.keys():
                        node_id = node_ids[id]
                        fl_graph.G.nodes[node_id]["entity"] = 2
                        tokens_e2_found += 1
                        entity2_dict[t.text] = fl_graph.G.nodes[node_id]
                    else:
                        e2_not_found.append(t.text)

            ud_graph_list.append(ud_graph.G)
            fl_graphs.append(fl_graph.G)

            # Build a report of missing or correctly mapped tokens for the FL graph
            reports.append(f"Entity 1: Found {tokens_e1_found} / {tokens_e1} token nodes\n" +
                           f"{entity1_dict}\n" +
                           f"Not found: {e1_not_found}\n\n" +
                           f"Entity 2: Found {tokens_e2_found} / {tokens_e2} token nodes\n" +
                           f"{entity2_dict}\n" +
                           f"Not found: {e2_not_found}\n\n")
            e1_found.append(0.0 if tokens_e1 == 0 else tokens_e1_found / tokens_e1)
            e2_found.append(0.0 if tokens_e2 == 0 else tokens_e2_found / tokens_e2)

        df_parsed = pd.DataFrame(
            {
                "SID": sentence_ids,
                "text": sentences,
                "label_id": label_ids,
                "ud": ud_graph_list,
                "fl": fl_graphs,
                "report_fl": reports,
                "e1_found_fl": e1_found,
                "e2_found_fl": e2_found
            }
        )

        return df_parsed

    @staticmethod
    def generate_dataframe_amr(df, sentence_colname="", label_id_colname="", sentence_id_colname=""):
        """
        Submit a dataframe and compute a response dataframe with entity-marked graphs
        Expects columns b1,e1 b2,e2 to provide begin/end character positions of the respective entities 1 and 2.
        :param df: The dataframe containing the input sentence and all necessary conversion information
        :param sentence_colname: which column is the sentence in
        :param label_id_colname: which column is the label in
        :param sentence_id_colname: which column is the unique sentence id in
        :return: a new dataframe with the above information, converted and entity-tagged AMR graphs as well as a
                 report for how many and which entity tokens could be mapped to their respective AMR nodes
        """
        def is_allcaps(text):
            return not re.search(r'[a-z]', text)

        # For each graph keep track of which entity tokens are linked to nodes, and which couldn't be found
        rows = df.iterrows()
        sentences = [row[1][sentence_colname] for row in rows]
        rows = df.iterrows()
        label_ids = [row[1][label_id_colname] for row in rows]
        rows = df.iterrows()
        sentence_ids = [row[1][sentence_id_colname] for row in rows]

        amr_graph_list = []
        reports = []
        e1_found = []
        e2_found = []

        # For my conversions I have been using the amrlib 0.8.0 xfm bart large model
        amr_stog = amrlib.load_stog_model()
        # Load the same spacy model that TUW NLP uses to build the graphs to get a comparable mapping
        spacy_nlp = spacy.load('en_core_web_sm')

        for i, sent in enumerate(sentences):

            # For each graph keep track of which entity tokens are linked to nodes, and which couldn't be found
            entity1_dict = {}
            entity2_dict = {}
            tokens_e1 = 0
            tokens_e1_found = 0
            tokens_e2 = 0
            tokens_e2_found = 0
            e1_not_found = []
            e2_not_found = []

            # Parse UD
            pn_graphs = amr_stog.parse_sents([sent])
            amr_graph = AMRGraph(pn_graphs[0], sent)

            # Map the nodes to their respective tokens
            token_to_node = {}
            for idx in amr_graph.G.nodes:
                t_id = amr_graph.G.nodes[idx]['token_id']
                if t_id is not None:
                    token_to_node[t_id] = idx

            # Unlike UD we don't get a neat mapping between token ID and the start/end characters.
            # So we redo the original spacy AMR conversion to get the token character info from there.
            doc = spacy_nlp(sent)
            indices = [(t.idx, t.idx + len(t)) for t in doc]

            tokens = json.loads(amr_graph.tokens)

            # Go through the tokens and check for each token if it is an entity. If yes, mark the node.
            for token_num, character_idx in enumerate(indices):
                start_char = character_idx[0]
                end_char = character_idx[1]
                tok = tokens[token_num]
                if (start_char >= df.iloc[i].b1 and end_char <= df.iloc[i].e1) \
                        or (df.iloc[i].b1 <= start_char <= df.iloc[i].e1 and is_allcaps(tok)) \
                        or (df.iloc[i].b1 <= end_char <= df.iloc[i].e1 and is_allcaps(tok)):
                    tokens_e1 += 1
                    if token_num in token_to_node.keys():
                        node_id = token_to_node[token_num]
                        amr_graph.G.nodes[node_id]["entity"] = 1
                        tokens_e1_found += 1
                        entity1_dict[tokens[token_num]] = amr_graph.G.nodes[node_id]
                    else:
                        e1_not_found.append(tokens[token_num])
                if (start_char >= df.iloc[i].b2 and end_char <= df.iloc[i].e2) \
                        or (df.iloc[i].b2 <= start_char <= df.iloc[i].e2 and is_allcaps(tok)) \
                        or (df.iloc[i].b2 <= end_char <= df.iloc[i].e2 and is_allcaps(tok)):
                    tokens_e2 += 1
                    if token_num in token_to_node.keys():
                        node_id = token_to_node[token_num]
                        amr_graph.G.nodes[node_id]["entity"] = 2
                        tokens_e2_found += 1
                        entity2_dict[tokens[token_num]] = amr_graph.G.nodes[node_id]
                    else:
                        e2_not_found.append(tokens[token_num])

            amr_graph_list.append(amr_graph.G)

            reports.append(f"Entity 1: Found {tokens_e1_found} / {tokens_e1} token nodes\n" +
                           f"{entity1_dict}\n" +
                           f"Not found: {e1_not_found}\n\n" +
                           f"Entity 2: Found {tokens_e2_found} / {tokens_e2} token nodes\n" +
                           f"{entity2_dict}\n" +
                           f"Not found: {e2_not_found}\n\n")
            e1_found.append(0.0 if tokens_e1 == 0 else tokens_e1_found / tokens_e1)
            e2_found.append(0.0 if tokens_e2 == 0 else tokens_e2_found / tokens_e2)

        df_parsed = pd.DataFrame(
            {
                "SID": sentence_ids,
                "text": sentences,
                "label_id": label_ids,
                "amr": amr_graph_list,
                "report_amr": reports,
                "e1_found_amr": e1_found,
                "e2_found_amr": e2_found
            }
        )

        return df_parsed

    def prune_graphs(self, graphs: List[nx.DiGraph] = None) -> None:
        graphs_str = []
        for i, graph in enumerate(graphs):
            graph.remove_nodes_from(list(nx.isolates(graph)))
            # ADAM: THIS IS JUST FOR PICKLE TO PENMAN CONVERSION
            graph = self._random_postprocess(graph)

            g = [
                c
                for c in sorted(
                    nx.weakly_connected_components(graph), key=len, reverse=True
                )
            ]
            if len(g) > 1:
                print(
                    "WARNING: graph has multiple connected components, taking the largest"
                )
                g_pn = graph_to_pn(graph.subgraph(g[0].copy()))
            else:
                g_pn = graph_to_pn(graph)

            graphs_str.append(g_pn)

        return graphs_str

    def read_dataset(
        self,
        examples: List[Tuple[str, str]] = None,
        path: str = None,
        binary: bool = False,
    ) -> List[Sample]:
        if examples:
            return [Sample(example, PotatoGraph()) for example in examples]
        elif path:
            if binary:
                df = pd.read_pickle(path)
                graphs = [graph for graph in df["graph"].tolist()]
                if type(graphs[0]) == str and check_if_str_is_penman(graphs[0]):
                    graphs_str = self.prune_graphs(df.graph.tolist())
                    df.drop(columns=["graph"], inplace=True)
                    df["graph"] = graphs_str
            else:
                df = pd.read_csv(path, sep="\t")

            return [
                Sample(
                    (example["text"], example["label"]),
                    potato_graph=PotatoGraph(graph=example["graph"]),
                    label_id=example["label_id"],
                )
                for _, example in tqdm(df.iterrows())
            ]
        else:
            raise ValueError("No examples or path provided")

    # ADAM: THIS WILL NEED TO BE ADDRESSED
    def _random_postprocess(self, graph: nx.DiGraph) -> nx.DiGraph:
        for node, attr in graph.nodes(data=True):
            if len(attr["name"].split()) > 1:
                attr["name"] = attr["name"].split()[0]

        return graph

    def to_dataframe(
        self, as_penman: bool = False, as_json: bool = False
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "text": [sample.text for sample in self._dataset],
                "label": [sample.label for sample in self._dataset],
                "label_id": [
                    sample.get_label_id(self.label_vocab) for sample in self._dataset
                ],
                "graph": [
                    str(sample.potato_graph)
                    if as_penman
                    else sample.potato_graph.to_dict()
                    if as_json
                    else sample.potato_graph.graph.G
                    for sample in self._dataset
                ],
            }
        )
        return df

    def parse_graphs(self, graph_format: str = "fourlang") -> List[PotatoGraph]:
        graphs = list(
            self.extractor.parse_iterable(
                [sample.text for sample in self._dataset], graph_format
            )
        )

        self.graphs = [PotatoGraph(graph=graph) for graph in graphs]
        return self.graphs

    def set_graphs(self, graphs: List[PotatoGraph]) -> None:
        for sample, potato_graph in zip(self._dataset, graphs):
            potato_graph.graph.G.remove_edges_from(
                nx.selfloop_edges(potato_graph.graph.G)
            )
            sample.set_graph(potato_graph)

    def load_graphs(self, path: str, binary: bool = False) -> None:
        if binary:
            graphs = [graph for graph in pd.read_pickle(path)]
            graph_str = self.prune_graphs(graphs)

            graphs = [PotatoGraph(graph=graph) for graph in graph_str]
            self.graphs = graphs
        else:
            with open(path, "rb") as f:
                for line in f:
                    graph = PotatoGraph(graph=line.strip())
                    self.graphs.append(graph)

        self.set_graphs(self.graphs)

    def save_dataset(self, path: str) -> None:
        df = self.to_dataframe()
        self.save_dataframe(df, path)

    def save_graphs(self, path: str, type="dict") -> None:
        with open(path, "wb") as f:
            for graph in self.graphs:
                if type == "dict":
                    json.dump(graph.graph, f)
                    f.write("\n")
                elif type == "penman":
                    f.write(f"{str(graph)}\n")
