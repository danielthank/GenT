import csv
import json
import os

from typing import Dict, List, Union
from ml.app_normalizer import extract_metadata, get_name
from paper.baseline.app_baseline_utils import remember_component_fields, EMPTY


def extract_rows_from_transaction(
    transaction: dict
) -> List[List[Union[str, int]]]:
    """
    Each transaction is being translated to a few rows, based on the depth parameter.
    For depth=1, each edge will be translated to a row.
    For depth=2, each concatenation of two edges will be translated to a row.
    Each row will have the following columns:
    - traceId
    - startTime
    - endTime
    - componentName
    - parentComponentName
    - hasError
    - ... 2 string features
    - ... 3 int features
    """
    transaction_data = []

    nodes: Dict[str, dict] = transaction["nodesData"]
    child_to_parent: Dict[str, dict] = {
        n["target"]: nodes[n["source"]] for n in transaction["graph"]["edges"]
    }
    for node_id, node in nodes.items():
        parent = child_to_parent.get(node_id)
        if not parent:
            continue

        int_features, string_features = extract_metadata(node)
        string_features = string_features[:2]
        int_features = int_features[:3]
        for i in range(len(string_features), 2):
            string_features.append(
                (
                    f"empty_string_feature_{i}",
                    "str",
                    EMPTY,
                )
            )
        for i in range(len(int_features), 3):
            int_features.append(
                (
                    f"empty_int_feature_{i}",
                    "int",
                    0,
                )
            )
        row = [
            int(transaction["details"]["transactionId"], 16),
            node["startTime"],
            node["startTime"] + node["duration"],
            get_name(nodes, node_id),
            get_name(nodes, parent["id"]),
            1 if node["issues"] else 0,
            *[feature[2] for feature in string_features],
            *[feature[2] for feature in int_features],
        ]
        remember_component_fields(
            get_name(nodes, node_id),
            [feature[0] for feature in string_features] + [feature[0] for feature in int_features]
        )
        transaction_data.append(row)

    return transaction_data


def get_csv_headers() -> List[str]:
    return [
        "traceId",
        "startTime",
        "endTime",
        "componentName",
        "parentComponentName",
        "hasError",
        "str_feature_1",
        "str_feature_2",
        "int_feature_1",
        "int_feature_2",
        "int_feature_3",
    ]


def normalize_data_baseline(input_dir: str, target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        with open(os.path.join(target_dir, file.replace(".json", ".csv")), "w") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(get_csv_headers())
            with open(os.path.join(input_dir, file), "r") as input_file:
                for json_line in input_file.readlines():
                    tx = json.loads(json_line)
                    for row in extract_rows_from_transaction(tx):
                        writer.writerow(row)
