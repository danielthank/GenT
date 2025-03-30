import datetime
import json
import os

import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from functools import lru_cache
from typing import List, Tuple, Dict
from matplotlib import pyplot

from ml.app_normalizer import extract_rows_from_transaction, get_csv_headers, set_gent_name
from ml.app_utils import GenTConfig, store_global_metadata

ALL_TRACES = 23010
GRAPH_COUNTS: List[Tuple[str, int]] = [
    ["[('fastapi-edge', 'api.twilio.com'), ('fastapi-edge', 'getSharedUnicorn'), ('getSharedUnicorn', 'sharedUnicorns'), ('getSharedUnicorn', 'wildrydes-prod-unicornDispatched'), ('wildrydes-prod-unicornDispatched', 'wildrydes-prod-recordRide'), ('wildrydes-prod-unicornDispatched', 'wildrydes-prod-uploadReceipt')]", 737],
    ["[('/prod/ride', 'wildrydes-prod-requestUnicorn'), ('socialprod', 'wildrydes-prod-postToSocial'), ('wildrydes-prod-PaymentRecords', 'wildrydes-prod-sendReceipt'), ('wildrydes-prod-Rides-144HA57HKYVE6', 'wildrydes-prod-sumRides'), ('wildrydes-prod-RyderQueue-rvOTc88cl7pn', 'wildrydes-prod-postToAnalytics'), ('wildrydes-prod-recordRide', 'wildrydes-prod-PaymentRecords'), ('wildrydes-prod-recordRide', 'wildrydes-prod-Rides-144HA57HKYVE6'), ('wildrydes-prod-requestUnicorn', 'socialprod'), ('wildrydes-prod-requestUnicorn', 'wildrydes-prod-OccupiedUnicorns-ZC0TEIXM6XLY'), ('wildrydes-prod-requestUnicorn', 'wildrydes-prod-unicornDispatched'), ('wildrydes-prod-requestUnicorn', 'wildrydes-prod-unicornMetric'), ('wildrydes-prod-sendReceipt', 'api.twilio.com'), ('wildrydes-prod-sumRides', 'wildrydes-prod-UnicornStats-18GOCZ26SLUAB'), ('wildrydes-prod-unicornDispatched', 'wildrydes-prod-recordRide'), ('wildrydes-prod-unicornDispatched', 'wildrydes-prod-uploadReceipt'), ('wildrydes-prod-unicornMetric', 'wildrydes-prod-RyderQueue-rvOTc88cl7pn'), ('wildrydes-prod-uploadReceipt', 'wildrydes-prod-ridereceipts-7vvop0svyhs9.s3.us-west-2.amazonaws.com')]", 78],
    ["[('eventBridge', 'wildrydes-prod-calcSalaries'), ('wildrydes-prod-calcSalaries', '4fsay0n12a.execute-api.us-east-1.amazonaws.com'), ('wildrydes-prod-calcSalaries', 'wildrydes-prod-UnicornStats-18GOCZ26SLUAB')]", 79],
    ["[('/prod/ride', 'wildrydes-prod-requestUnicorn'), ('socialprod', 'wildrydes-prod-postToSocial'), ('wildrydes-prod-PaymentRecords', 'wildrydes-prod-sendReceipt'), ('wildrydes-prod-Rides-144HA57HKYVE6', 'wildrydes-prod-sumRides'), ('wildrydes-prod-RyderQueue-rvOTc88cl7pn', 'wildrydes-prod-postToAnalytics'), ('wildrydes-prod-recordRide', 'wildrydes-prod-PaymentRecords'), ('wildrydes-prod-recordRide', 'wildrydes-prod-Rides-144HA57HKYVE6'), ('wildrydes-prod-requestUnicorn', 'socialprod'), ('wildrydes-prod-requestUnicorn', 'wildrydes-prod-OccupiedUnicorns-ZC0TEIXM6XLY'), ('wildrydes-prod-requestUnicorn', 'wildrydes-prod-unicornDispatched'), ('wildrydes-prod-requestUnicorn', 'wildrydes-prod-unicornMetric'), ('wildrydes-prod-sendReceipt', 'api.twilio.com'), ('wildrydes-prod-unicornDispatched', 'wildrydes-prod-recordRide'), ('wildrydes-prod-unicornDispatched', 'wildrydes-prod-uploadReceipt'), ('wildrydes-prod-unicornMetric', 'wildrydes-prod-RyderQueue-rvOTc88cl7pn'), ('wildrydes-prod-uploadReceipt', 'wildrydes-prod-ridereceipts-7vvop0svyhs9.s3.us-west-2.amazonaws.com')]", 36],
    ["[('/prod/purchase', 'wildrydes-prod-purchaseNewUnicron'), ('wildrydes-prod-purchaseNewUnicron', 'Redis'), ('wildrydes-prod-purchaseNewUnicron', 'admin'), ('wildrydes-prod-purchaseNewUnicron', 'api.twilio.com'), ('wildrydes-prod-purchaseNewUnicron', 'jgeegotidbetxnvd4mghwah3ae.appsync-api.us-west-2.amazonaws.com'), ('wildrydes-prod-purchaseNewUnicron', 'q5bei6rdyfdwzgofnmld56h4pe.appsync-api.us-west-2.amazonaws.com'), ('wildrydes-prod-purchaseNewUnicron', 'wildrydes.unicrons')]", 21],
    ["[('wildrydes-prod-PaymentRecords', 'wildrydes-prod-sendReceipt'), ('wildrydes-prod-sendReceipt', 'api.twilio.com')]", 23],
    ["[('/prod/purchase', 'wildrydes-prod-purchaseNewUnicron'), ('wildrydes-prod-purchaseNewUnicron', 'Redis'), ('wildrydes-prod-purchaseNewUnicron', 'api.twilio.com'), ('wildrydes-prod-purchaseNewUnicron', 'jgeegotidbetxnvd4mghwah3ae.appsync-api.us-west-2.amazonaws.com'), ('wildrydes-prod-purchaseNewUnicron', 'q5bei6rdyfdwzgofnmld56h4pe.appsync-api.us-west-2.amazonaws.com'), ('wildrydes-prod-purchaseNewUnicron', 'wildrydes.unicrons')]", 8],
    ["[('/prod/purchase', 'wildrydes-prod-purchaseNewUnicron'), ('wildrydes-prod-purchaseNewUnicron', 'Redis'), ('wildrydes-prod-purchaseNewUnicron', 'admin'), ('wildrydes-prod-purchaseNewUnicron', 'jgeegotidbetxnvd4mghwah3ae.appsync-api.us-west-2.amazonaws.com'), ('wildrydes-prod-purchaseNewUnicron', 'q5bei6rdyfdwzgofnmld56h4pe.appsync-api.us-west-2.amazonaws.com'), ('wildrydes-prod-purchaseNewUnicron', 'wildrydes.unicrons')]", 7],
    ["[('getSharedUnicorn', 'sharedUnicorns'), ('getSharedUnicorn', 'wildrydes-prod-unicornDispatched'), ('wildrydes-prod-unicornDispatched', 'wildrydes-prod-recordRide'), ('wildrydes-prod-unicornDispatched', 'wildrydes-prod-uploadReceipt')]", 6],
    ["[('/prod/purchase', 'wildrydes-prod-purchaseNewUnicron'), ('wildrydes-prod-purchaseNewUnicron', 'Redis'), ('wildrydes-prod-purchaseNewUnicron', 'admin'), ('wildrydes-prod-purchaseNewUnicron', 'api.twilio.com'), ('wildrydes-prod-purchaseNewUnicron', 'jgeegotidbetxnvd4mghwah3ae.appsync-api.us-west-2.amazonaws.com'), ('wildrydes-prod-purchaseNewUnicron', 'q5bei6rdyfdwzgofnmld56h4pe.appsync-api.us-west-2.amazonaws.com'), ('wildrydes-prod-purchaseNewUnicron', 'wildrydes.free.beeceptor.com'), ('wildrydes-prod-purchaseNewUnicron', 'wildrydes.unicrons')]", 2],
    ["[('/prod/purchase', 'wildrydes-prod-purchaseNewUnicron'), ('wildrydes-prod-purchaseNewUnicron', 'Redis'), ('wildrydes-prod-purchaseNewUnicron', 'jgeegotidbetxnvd4mghwah3ae.appsync-api.us-west-2.amazonaws.com'), ('wildrydes-prod-purchaseNewUnicron', 'q5bei6rdyfdwzgofnmld56h4pe.appsync-api.us-west-2.amazonaws.com'), ('wildrydes-prod-purchaseNewUnicron', 'wildrydes.unicrons')]", 3]
]

# Configurations of the adaption experiment
BULK_SIZE = 5000
CONFIG_BASE = GenTConfig(chain_length=2, iterations=5, tx_start=0, tx_end=BULK_SIZE)
CONFIG_ERROR = GenTConfig(chain_length=2, iterations=5, tx_start=BULK_SIZE, tx_end=2 * BULK_SIZE)
CONFIG_RATE = GenTConfig(chain_length=2, iterations=5, tx_start=2 * BULK_SIZE, tx_end=3 * BULK_SIZE)
CONFIG_LESS_ERRORS = GenTConfig(chain_length=2, iterations=5, tx_start=3 * BULK_SIZE, tx_end=4 * BULK_SIZE)
ROLLING_EXPERIMENT_NAMES = ["base", "error", "rate", "less_errors"]
ROLLING_EXPERIMENT_CONFIGS = [CONFIG_BASE, CONFIG_ERROR, CONFIG_RATE, CONFIG_LESS_ERRORS]


def edge_index_to_graph(edge_index: torch.Tensor, index_to_node: Dict[int, str]) -> str:
    return str([(index_to_node[edge_index[0][i].item()], index_to_node[edge_index[1][i].item()]) for i in range(edge_index.shape[1])])


def get_adaption_experiment_txs(tx_start: int, tx_end: int, all_txs: List[dict]) -> List[dict]:
    if tx_start == CONFIG_ERROR.tx_start and tx_end == CONFIG_ERROR.tx_end:
        # Add issues to every second tx
        for i in range(len(all_txs)):
            if i % 2 == 0:
                continue
            entry_point = sorted(list(all_txs[i]["nodesData"].values()), key=lambda n: n["startTime"])[0]
            entry_point["issues"] = [{"name": "AdaptionError"}]
        return all_txs
    elif tx_start == CONFIG_RATE.tx_start and tx_end == CONFIG_RATE.tx_end:
        # duplicate every tx
        for i in range(len(all_txs)):
            tx = all_txs[i]
            new_tx = json.loads(json.dumps(tx))
            new_tx["details"]["startTime"] += 100
            new_tx["details"]["transactionId"] += '_dup'
            new_nodes_data = {}
            for node in new_tx["nodesData"].values():
                node["startTime"] += 100
                node["id"] += '_dup'
                new_nodes_data[node["id"]] = node
            new_tx["nodesData"] = new_nodes_data
            new_tx["graph"]["edges"] = [{"source": e["source"] + '_dup', "target": e["target"] + '_dup'} for e in new_tx["graph"]["edges"]]
            all_txs.append(new_tx)
        return all_txs
    elif tx_start == CONFIG_LESS_ERRORS.tx_start and tx_end == CONFIG_LESS_ERRORS.tx_end:
        # Add issues to every second tx
        for i in range(len(all_txs)):
            if i % 2 == 0:
                continue
            for node in all_txs[i]["nodesData"].values():
                node["issues"] = []
        return all_txs
    return all_txs


@lru_cache(maxsize=10)
def get_all_txs(tx_start: int, tx_end: int, traces_dir: str = TRACES_DIR) -> List[dict]:
    txs = []
    for file in os.listdir(traces_dir):
        with open(os.path.join(traces_dir, file), "r") as input_file:
            for json_line in input_file.readlines():
                tx = json.loads(json_line.rstrip(',\n'))
                if not tx["graph"]["edges"] or not all(e["target"] for e in tx["graph"]["edges"]):
                    continue
                set_gent_name(tx["nodesData"])
                txs.append(tx)
    result = sorted(txs, key=lambda t: t["details"]["startTime"])[tx_start: tx_end]
    if tx_end != -1 and len(result) != tx_end - tx_start:
        print(f"Warning: Not enough txs, using {len(result)} instead of {tx_end - tx_start}")
    result = get_adaption_experiment_txs(tx_start, tx_end, result)
    return result


@lru_cache(maxsize=10)
def get_full_dataset_chains(config: GenTConfig, load_all: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    relevant_txs = get_all_txs(0, ALL_TRACES, config.traces_dir) if load_all else get_all_txs(config.tx_start, config.tx_end, config.traces_dir)
    all_parsed_txs = []
    subset_parsed_txs = []
    for tx_index, tx in enumerate(relevant_txs):
        rows = extract_rows_from_transaction(tx, config, tag_root_chains=True)
        edges = sorted({(tx["nodesData"][n["source"]]["gent_name"],
                         tx["nodesData"][n["target"]]["gent_name"]) for n in tx["graph"]["edges"]})
        rows = [[str(edges)] + row[1:] for row in rows]  # replace traceId with graph encoding
        all_parsed_txs.extend(rows)
        if config.tx_start <= tx_index < config.tx_end:
            subset_parsed_txs.extend(rows)
    column_names = ["graph"] + get_csv_headers(config)[1:] + ["is_root_chain"]
    all_df = pd.DataFrame(all_parsed_txs, columns=column_names)
    subset_df = pd.DataFrame(subset_parsed_txs, columns=column_names)
    store_global_metadata(config)
    return subset_df, all_df


@lru_cache(maxsize=1000)
def get_graph_counts(traces_dir: str, tx_start: int = 0, tx_end: int = 2 ** 32) -> Dict[str, int]:
    txs = get_all_txs(tx_start, tx_end, traces_dir)
    graph_counts = defaultdict(int)
    for tx in txs:
        edges = sorted({(tx["nodesData"][n["source"]]["gent_name"], tx["nodesData"][n["target"]]["gent_name"]) for n in tx["graph"]["edges"]})
        graph_counts[str(edges)] += 1
    return graph_counts


def count_histogram(traces_dir: str):
    txs = get_all_txs(0, 2 ** 32, traces_dir)
    tx_times = [datetime.datetime.fromtimestamp(tx["details"]["startTime"] / 1000) for tx in txs]
    df = pd.DataFrame({"datetime": tx_times})
    fig, ax = pyplot.subplots()
    df["datetime"].astype(np.int64).plot.hist(ax=ax, bins=100)
    ax.set_xticklabels(pd.to_datetime(ax.get_xticks().tolist()), rotation=60)
    fig.tight_layout()
    pyplot.show()


if __name__ == '__main__':
    # TODO: pass traces_dir
    counts = get_graph_counts(tx_start=0, tx_end=ALL_TRACES, traces_dir="my_traces_dir")
    print(',\n'.join(str([k, v]) for k, v in counts.items()))
    counts = json.dumps(counts)
    print("raw size", len(counts))
    import gzip
    print("gzip size", len(gzip.compress(counts.encode("utf-8"))))
    # count_histogram()
