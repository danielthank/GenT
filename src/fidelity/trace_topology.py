import json
import os
from typing import Dict, List, Tuple, Set

import numpy as np
from zss import Node, simple_distance

from drivers.base_driver import BaseDriver
from fidelity.utils import ComparableForest, ComparableTree, StructureId, \
    build_comparable_forest_of_subgraphs, get_transactions_graphs_generator, TRANSACTION_RAW_DIR
from ml.app_utils import GenTBaseConfig


def init_counts(raw_forest: ComparableForest, generated_forest: ComparableForest)\
        -> Tuple[Dict[str, int], Dict[str, int]]:
    generated_forest_counts: Dict[str, int] = {}
    raw_forest_counts: Dict[str, int] = {}
    for raw_id, raw_tree in raw_forest.forest.items():
        raw_forest_counts[raw_id] = raw_tree.appearances
    for generated_id, generated_tree in generated_forest.forest.items():
        generated_forest_counts[generated_id] = generated_tree.appearances
    return raw_forest_counts, generated_forest_counts


def reduce_forests_by_id(
    raw_id: str,
    raw_forest_counts: Dict[str, int],
    generated_id: str,
    generated_forest_counts: Dict[str, int],
) -> int:
    """
    This function count (and remove) the trees with the given ids.
    """
    total_matches = 0
    raw_tree = raw_forest_counts[raw_id]
    generated_tree = generated_forest_counts[generated_id]
    if raw_tree > generated_tree:
        generated_forest_counts.pop(generated_id)
        total_matches += generated_tree
        raw_forest_counts[raw_id] -= - generated_tree

    elif raw_tree < generated_tree:
        raw_forest_counts.pop(raw_id)
        total_matches += raw_tree
        generated_forest_counts[generated_id] -= raw_tree
    else:
        raw_forest_counts.pop(raw_id)
        generated_forest_counts.pop(generated_id)
        total_matches += raw_tree
    return total_matches


def compare_topologies(topo1: ComparableTree, topo2: ComparableTree):
    def build_tree(topo: ComparableTree) -> Node:
        graph = topo.graphs[0]
        id_to_node: Dict[str, Node] = {}
        children: Set[str] = set()
        for edge in graph.edges:
            id_to_node[edge[0]] = Node(edge[0])
            id_to_node[edge[1]] = Node(edge[1])
        for edge in graph.edges:
            children.add(edge[1])
        root_node = next(
            (node for node_id, node in id_to_node.items() if node_id not in children),
            None,
        )
        if not root_node:
            #  There is a cycle
            root_node = next(iter(id_to_node.values()))

        for edge in graph.edges:
            if root_node == id_to_node[edge[1]]:
                #  There is a cycle, let's break it in the random root. Note: there are cases where it will fail
                continue
            id_to_node[edge[0]].addkid(id_to_node[edge[1]])
        return root_node

    return int(simple_distance(build_tree(topo1), build_tree(topo2)))


def compute_match_score(
    tree: ComparableTree, forest: ComparableForest
) -> Dict[StructureId, int]:
    """
    This function compute the match score of a tree to all other trees in a forest.
    High score means high diff (the tree is less similar to the other tree).
    """
    score: Dict[StructureId, int] = {}
    for structure_id, other_tree in forest.forest.items():
        score[structure_id] = compare_topologies(tree, other_tree)
    return score


def get_greedy_match_distances(
    raw_forest: ComparableForest, generated_forest: ComparableForest
) -> List[int]:
    """
    This function compute the match score of all trees in the raw forest to all trees in the generated forest.
    """
    raw_forest_counts, generated_forest_counts = init_counts(raw_forest, generated_forest)
    if sum(raw_forest_counts.values()) != sum(generated_forest_counts.values()):
        raise ValueError("The number of trees in the two forests is not equal")
    all_scores: List[int] = []
    structure_ids_to_score: Dict[Tuple[StructureId, StructureId], int] = {}
    for generated_id, tree in generated_forest.forest.items():
        score = compute_match_score(tree, raw_forest)
        structure_ids_to_score.update(
            {
                (raw_id, generated_id): match_score
                for raw_id, match_score in score.items()
            }
        )
    sorted_score = sorted(structure_ids_to_score, key=structure_ids_to_score.get)
    for raw_id, generated_id in sorted_score:
        if (
            raw_id not in raw_forest_counts
            or generated_id not in generated_forest_counts
        ):
            # We already removed this trees with a better match
            continue
        match_score = structure_ids_to_score[(raw_id, generated_id)]
        number_of_trees = reduce_forests_by_id(
            raw_id, raw_forest_counts, generated_id, generated_forest_counts
        )
        all_scores += [match_score] * number_of_trees
    return all_scores


def get_forest_score(driver: BaseDriver, subtree_height: int = 2) -> Dict[str, float]:
    target_path = driver.forest_results_path(subtree_height)
    if os.path.exists(target_path):
        return json.load(open(target_path))
    generated_forest = build_comparable_forest_of_subgraphs(
        get_transactions_graphs_generator(driver.get_generated_data_folder()),
        subtree_height=subtree_height,
        max_trees=50_000
    )
    raw_forest = build_comparable_forest_of_subgraphs(
        get_transactions_graphs_generator(TRANSACTION_RAW_DIR),
        subtree_height=subtree_height,
        max_trees=generated_forest.size()
    )
    total_score = get_greedy_match_distances(raw_forest, generated_forest)
    data = {
        'average': float(np.average(total_score)),
        'std': float(np.std(total_score)),
    }
    with open(target_path, 'w') as f:
        json.dump(data, f)
    return data


if __name__ == '__main__':
    for height in [2, 3, 5, 7, 9]:
        print(height, get_forest_score(
            driver=NetShareDriver(GenTBaseConfig(
                chain_length=3, iterations=100, metadata_str_size=1, metadata_int_size=2
            )),
            subtree_height=height,
        ))
