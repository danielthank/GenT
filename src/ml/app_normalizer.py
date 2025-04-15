import csv
import json
import math
import os
import argparse
from typing import Dict, Generator, List, Optional, Tuple, Union, NamedTuple

RowType = List[Union[str, int]]

from ml.app_utils import (
    EMPTY,
    MetadataType,
    GenTConfig,
    get_features,
    get_metadata_size,
    remember_type,
    store_global_metadata,
)
INITIAL_ROW_HEADERS = ["traceId", "txStartTime", "chain"]
PER_COMPONENT_HEADERS = [
    "gapFromParent_{component_index}",
    "duration_{component_index}",
    "hasError_{component_index}",
]
METADATA_HEADER = "metadata_{component_index}_{metadata_index}"

all_chains = set()
all_nodes = set()


def get_time(n: dict, tx_start_time: int) -> int:
    return int(n.get("startTime")) or tx_start_time  # type: ignore


def get_name(nodes_data: dict, node_id: str) -> str:
    if "gent_name" in nodes_data[node_id]:
        return nodes_data[node_id]["gent_name"]
    name = str(nodes_data[node_id]["resource"]["name"])
    dup_names = [k for k, n in nodes_data.items() if n["resource"]["name"] == name]
    if name.startswith("POST") or name.startswith("GET"):
        name = name.split(" ")[1]
    # elif "/" in name:
    #     name = name.split("/")[0] or name
    dup_names = sorted(dup_names, key=lambda k: nodes_data[k]["startTime"])
    name += f"*{dup_names.index(node_id)}"
    return name


def set_gent_name(nodes_data: dict):
    for node_id, node in nodes_data.items():
        if "gent_name" not in node:
            node["gent_name"] = get_name(nodes_data, node_id)


"""
This format normalizer reads traces files that are app dumped file,
    and converts the given values in the given keys to canonical format.

The created format has the following columns:
- startTime
- endTime
- traceId
- componentName
- parentComponentName
- hasError
"""


def extract_metadata(
    node: dict,
) -> Tuple[List[Tuple[str, MetadataType, int]], List[Tuple[str, MetadataType, str]]]:
    int_features: List[Tuple[str, MetadataType, int]] = []
    string_features: List[Tuple[str, MetadataType, str]] = []
    service_type = node["resource"]["serviceType"]
    remember_type(node["gent_name"], service_type)
    if service_type == "lambda":
        string_features.append(("coldStart", "str", str(node["coldStart"])))
        string_features.append(("runtime", "str", node["resource"]["runtime"]))
        string_features.append(
            ("issue.name", "str", (node["issues"] or [{}])[0].get("name") or EMPTY)
        )
        string_features.append(
            (
                "event.detail-type",
                "str",
                node["event"].get("body", {}).get("detail-type") or EMPTY,
            )
        )
        string_features.append(
            (
                "event.source",
                "str",
                node["event"].get("body", {}).get("source") or EMPTY,
            )
        )
        int_features.append(
            ("event.size", "int", len(json.dumps(node["event"]["body"])))
        )
        int_features.append(("len_issues", "int", len(node["issues"] or [])))
        int_features.append(("memoryUsed", "int", int(node["memory"]["avg"] or 0)))
        int_features.append(
            (
                "event.locationLatitude",
                "int",
                int(math.fabs(float(node["event"]["body"].get("locationLatitude", 0)))),
            )
        )
        int_features.append(
            (
                "event.locationLongitude",
                "int",
                int(
                    math.fabs(float(node["event"]["body"].get("locationLongitude", 0)))
                ),
            )
        )
    elif service_type == "ecs":
        string_features.append(
            (
                "executionTags.route",
                "str",
                [
                    tag["value"]
                    for tag in node["executionTags"]
                    if tag["key"] == "route"
                ][0].strip("/"),
            )
        )
        string_features.append(
            ("issueName", "str", (node.get("issues") or [{}])[0].get("name") or EMPTY)
        )
        string_features.append(("method", "str", node.get("method") or EMPTY))
        string_features.append(
            (
                "environmentVariables.APP_NAME",
                "str",
                node.get("environmentVariables", {}).get("body", {}).get("APP_NAME")
                or EMPTY,
            )
        )
        string_features.append(
            (
                "resource.runtime",
                "str",
                node.get("resource", {}).get("runtime") or EMPTY,
            )
        )

        string_features.append(("statusCode", "str", str(node["statusCode"] or EMPTY)))
        int_features.append(("len_executionTags", "int", len(node["executionTags"])))
        int_features.append(
            (
                "len_environmentVariables",
                "int",
                len(node.get("environmentVariables", {}).get("body", {})),
            )
        )
        int_features.append(("duration", "int", node["duration"]))
    elif node.get("type") == "http":
        req_headers = ((node.get("request") or {}).get("headers") or {}).get("body") or {}
        res_headers = ((node["response"] or {}).get("headers") or {}).get("body") or {}
        string_features.append(("http.method", "str", node["method"] or EMPTY))
        string_features.append(
            (
                "request.headers.user-agent",
                "str",
                (req_headers.get("user-agent") or EMPTY).split('/')[0],
            )
        )
        string_features.append(("statusCode", "str", str(node["statusCode"] or EMPTY)))
        int_features.append(
            (
                "request.headers.content-length",
                "int",
                int(
                    req_headers.get("Content-Length")
                    or req_headers.get("content-length")
                    or 0
                ),
            )
        )
        int_features.append(
            (
                "response.headers.content-length",
                "int",
                int(
                    res_headers.get("Content-Length")
                    or res_headers.get("content-length")
                    or 0
                ),
            )
        )
        int_features.append(("len_request.headers", "int", len(req_headers)))
        if service_type == "dynamodb":
            string_features.append(
                (
                    "request.TableName",
                    "str",
                    node["request"]["body"]["body"]["TableName"] or EMPTY,
                )
            )
            string_features.append(
                (
                    "request.headers.x-amz-target",
                    "str",
                    node["request"]["headers"]["body"].get("x-amz-target") or EMPTY,
                )
            )
            int_features.append(
                (
                    "request.len_ExpressionAttributeNames",
                    "int",
                    len(node["request"].get("body", {}).get("body", {}).get("ExpressionAttributeNames") or []),
                )
            )
            int_features.append(
                (
                    "response.Count",
                    "int",
                    int(((node["response"].get("body") or {}).get("body") or {}).get("Count") or 0),
                )
            )
    elif node.get("type") == "jaeger":
        tags = node["environmentVariables"]["body"]
        if 'http.method' in tags:
            string_features.append(("component", "str", tags.get("component") or EMPTY))
            string_features.append(("request.http.method", "str", tags.get("http.method") or EMPTY))
            string_features.append(("request.peer.address", "str", tags.get("peer.address") or EMPTY))
            int_features.append(("request.http.status_code", "int", tags.get("http.status_code") or 0))
            int_features.append(("request.http.num_parameters", "int", (tags.get("http.url") or "").count("&")))
        else:
            string_features.append(("process_hostname", "str", tags.get("process_hostname") or EMPTY))
            string_features.append(("process_ip", "str", tags.get("process_ip") or EMPTY))
    return int_features, string_features


def extract_node_features(
    node: dict, parent: Optional[dict], tx_start_time: int, config: GenTConfig
) -> RowType:
    start_time = get_time(node, tx_start_time)
    gap_from_parent = min(max(start_time - get_time(parent, tx_start_time) if parent else 0, 0), 5000)
    duration = node["duration"] if node["duration"] < 5000 else 0
    has_error = 1 if node["issues"] else 0
    int_features, string_features = extract_metadata(node)
    return [gap_from_parent, duration, has_error] + get_features(  # type: ignore
        node["gent_name"], int_features, string_features, config=config
    )


def get_chains(
    transaction: dict, child_to_parent: dict, config: GenTConfig
) -> Generator[List[dict], None, None]:
    """
    This method returns a generator of chains, where each chain is a list of nodes.
    Note that we learn each edge in the graph, and not the nodes, and each edge just once.
    Order the children by lexicographic order of their name.
    """
    nodes: Dict[str, dict] = transaction["nodesData"]
    outgoing_edges_cnt = {node_id: 0 for node_id in nodes.keys()}
    for parent in child_to_parent.values():
        outgoing_edges_cnt[parent] = outgoing_edges_cnt.get(parent, 0) + 1
    
    queue = [node_id for node_id, cnt in outgoing_edges_cnt.items() if cnt == 0]
    visited = set()

    while queue:
        node_id = queue.pop(0)
        if node_id in child_to_parent:
            parent = child_to_parent[node_id]
            outgoing_edges_cnt[parent] -= 1
            if outgoing_edges_cnt[parent] == 0:
                queue.append(parent)
        chain = []
        for _ in range(config.chain_length - 1):
            if node_id in visited:
                break
            visited.add(node_id)
            chain.append(nodes[node_id])
            node_id = child_to_parent.get(node_id)
            if node_id is None:
                break
        if node_id is not None:
            chain.append(nodes[node_id])
        if len(chain) == 1:
            continue
        # print("chain", [item["id"] for item in chain])
        # Processed from leaf to root so reverse the chain
        yield chain[::-1]
    # all_nodes.update(n["gent_name"] for n in chain)
    # all_chains.add("&".join(n["gent_name"] for n in chain))

def get_default_row(config: GenTConfig) -> RowType:
    return (
            [0] * len(PER_COMPONENT_HEADERS) +
            [EMPTY] * config.metadata_str_size +
            [0] * config.metadata_int_size
    )


def extract_rows_from_transaction(
    transaction: dict, config: GenTConfig, tag_root_chains: bool = False
) -> List[RowType]:
    """
    Each transaction is being translated to a few rows, based on the depth parameter.
    For depth=1, each edge will be translated to a row.
    For depth=2, each concatenation of two edges will be translated to a row.
    Each row will have the following columns:
    - traceId
    - startTime
    - chain
    - for every depth level `i`:
        - gapFromParent_i
        - duration_i
        - hasError_i
    """
    transaction_data = []

    nodes: Dict[str, dict] = transaction["nodesData"]
    child_to_parent: Dict[str, dict] = {
        e["target"]: e["source"] for e in transaction["graph"]["edges"]
    }
    root_nodes = set(nodes.keys()) - set(child_to_parent.keys())
    set_gent_name(nodes)
    tx_start_time = transaction["details"]["startTime"]
    if len(nodes) == 1:
        print("Found a transaction with no edges")
        return []
    processed_nodes = set()
    for chain in get_chains(transaction, child_to_parent, config=config):
        chain_id = "#".join(n["gent_name"] for n in chain)
        row = [
            int(transaction["details"]["transactionId"], 16),
            tx_start_time,
            chain_id,
        ]
        for node in chain:
            node_features = extract_node_features(
                node, nodes.get(child_to_parent.get(node["id"])), tx_start_time, config=config
            )
            row.extend(node_features)
        if len(chain) < config.chain_length:
            default_row = get_default_row(config)
            for _ in range(config.chain_length - len(chain)):
                row.extend(default_row)

        if tag_root_chains:
            row.append(chain[0]["id"] in root_nodes)
        processed_nodes.update(n["gent_name"] for n in chain)

        transaction_data.append(row)

    return transaction_data


def get_csv_headers(config: GenTConfig) -> List[str]:
    headers = INITIAL_ROW_HEADERS.copy()
    for component_index in range(config.chain_length):
        headers += [
            h.format(component_index=component_index) for h in PER_COMPONENT_HEADERS
        ]
        for metadata_index in range(get_metadata_size(config)):
            headers += [
                METADATA_HEADER.format(
                    component_index=component_index, metadata_index=metadata_index
                )
            ]
    return headers


def normalize_data(input_dir: str, config: GenTConfig) -> None:
    all_nodes.clear()
    all_chains.clear()
    os.makedirs(config.get_raw_normalized_data_dir(), exist_ok=True)
    for file in os.listdir(input_dir):
        with open(
            os.path.join(
                config.get_raw_normalized_data_dir(), file.replace(".json", ".csv")
            ),
            "w",
        ) as output_file:
            writer = csv.writer(output_file)
            writer.writerow(get_csv_headers(config))
            with open(os.path.join(input_dir, file), "r") as input_file:
                for json_line in input_file.readlines():
                    json_line = json_line.strip()[:-1]
                    tx = json.loads(json_line)
                    for row in extract_rows_from_transaction(tx, config=config):
                        writer.writerow(row)

    store_global_metadata(config)
    print(f"Found a total of {len(all_nodes)} nodes and {len(all_chains)} chains")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Normalizer')
    parser.add_argument('--traces_dir', type=str, required=True, help='Directory containing trace data')
    args = parser.parse_args()
    normalize_data(args.traces_dir, GenTConfig(chain_length=2, traces_dir=args.traces_dir))
    normalize_data(args.traces_dir, GenTConfig(chain_length=3, traces_dir=args.traces_dir))
    normalize_data(args.traces_dir, GenTConfig(chain_length=4, traces_dir=args.traces_dir))
