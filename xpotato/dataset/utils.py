import json
import re
from collections import defaultdict

import networkx as nx
import pandas as pd
import penman as pn
from networkx.readwrite import json_graph
from tuw_nlp.graph.utils import preprocess_node_alto


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    df_to_save = df.copy()
    graphs = [
        json.dumps(json_graph.adjacency_data(g)) if type(g) == nx.DiGraph else g
        for g in df["graph"].tolist()
    ]
    df_to_save["graph"] = graphs
    df_to_save.to_csv(path, index=False, sep="\t")


def ud_to_graph(sen, edge_attr="color"):
    """convert dependency-parsed stanza Sentence to nx.DiGraph"""
    G = nx.DiGraph()
    root_id = None
    for word in sen.to_dict():
        if isinstance(word["id"], (list, tuple)):
            # token representing an mwe, e.g. "vom" ~ "von dem"
            continue
        G.add_node(word["id"], name=preprocess_node_alto(word["lemma"]))
        if word["deprel"] == "root":
            root_id = word["id"]
            G.add_node(word["head"], name="root")
        G.add_edge(word["head"], word["id"])
        G[word["head"]][word["id"]].update(
            {edge_attr: preprocess_node_alto(word["deprel"])}
        )

    return G, root_id


def default_pn_to_graph(raw_dl, edge_attr="color"):
    g = pn.decode(raw_dl)
    G = nx.DiGraph()

    char_to_id = defaultdict(int)
    char_to_entity = dict()
    next_id = 0
    for i, trip in enumerate(g.triples):
        #print(f"Potato: {trip}")
        if i == 0:
            root_id = next_id
            name = trip[2]
            if not name:
                name = "None"
            G.add_node(root_id, name=name)
            char_to_id[trip[0]] = next_id
            next_id += 1

        elif re.match("entity", trip[0].split('_')[0]):
            char_to_entity[trip[0]] = trip[2]

        elif trip[1] == ":instance":
            if trip[2]:
                name = trip[2]
                if not name:
                    name = "None"
                G.add_node(next_id, name=name)
                char_to_id[trip[0]] = next_id
                next_id += 1

    for trip in g.triples:
        if re.match(r":entity", trip[1]):
            G.nodes[char_to_id[trip[0]]]["entity"] = int(char_to_entity[trip[2]])
        elif trip[1] != ":instance":
            edge = trip[1].split(":")[1]
            src = trip[0]
            tgt = trip[2]
            if src not in char_to_id:
                char_to_id[src] = next_id
                name = src
                if not name:
                    name = "None"
                G.add_node(next_id, name=name)
                next_id += 1
            if tgt not in char_to_id:
                char_to_id[tgt] = next_id
                name = tgt
                if not name:
                    name = "None"
                G.add_node(next_id, name=name)
                next_id += 1

            G.add_edge(char_to_id[src], char_to_id[tgt])
            G[char_to_id[src]][char_to_id[tgt]].update({edge_attr: edge})

    return G, root_id


def amr_pn_to_graph(raw_dl, edge_attr="color", clean_nodes=False):
    g = pn.decode(raw_dl)
    G = nx.DiGraph()

    char_to_id = defaultdict(int)
    next_id = 0
    for i, trip in enumerate(g.triples):
        if i == 0:
            root_id = next_id
            if "-" in trip[2]:
                name = "-".join(trip[2].split("-")[:-1])
            else:
                name = trip[2]
            if clean_nodes:
                if name:
                    name = preprocess_node_alto(name)
                else:
                    name = "None"
            G.add_node(root_id, name=name)
            char_to_id[trip[0]] = next_id
            next_id += 1

        elif trip[1] == ":instance":
            if trip[2]:
                if "-" in trip[2]:
                    name = "-".join(trip[2].split("-")[:-1])
                else:
                    name = trip[2]
                if clean_nodes:
                    if name:
                        name = preprocess_node_alto(name)
                    else:
                        name = "None"
                G.add_node(next_id, name=name)
                char_to_id[trip[0]] = next_id
                next_id += 1

    for trip in g.triples:
        if trip[1] != ":instance":
            edge = "".join(trip[1].split(":")[1].split("-"))
            src = trip[0]
            tgt = trip[2]
            if src not in char_to_id:
                char_to_id[src] = next_id
                if "-" in src:
                    name = "-".join(src.split("-")[:-1])
                else:
                    name = src
                if clean_nodes:
                    if name:
                        name = preprocess_node_alto(name)
                    else:
                        name = "None"
                G.add_node(next_id, name=name)
                next_id += 1
            if tgt not in char_to_id:
                char_to_id[tgt] = next_id
                if "-" in tgt:
                    name = "-".join(tgt.split("-")[:-1])
                else:
                    name = tgt
                if clean_nodes:
                    if name:
                        name = preprocess_node_alto(name)
                    else:
                        name = "None"
                G.add_node(next_id, name=name)
                next_id += 1

            G.add_edge(char_to_id[src], char_to_id[tgt])
            G[char_to_id[src]][char_to_id[tgt]].update({edge_attr: edge})

    return G, root_id
