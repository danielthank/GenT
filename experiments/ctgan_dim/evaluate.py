import os
import shutil
import argparse
import sqlite3
import json
from drivers.gent.data import get_all_txs
from ml.app_normalizer import extract_metadata
from fidelity.tasks import trigger_correlation, relative_duration

ALL_TRACES = 9342

def create_spans_table(conn: sqlite3.Connection, table_name: str):
    cursor = conn.cursor()
    # Drop the table if it exists
    cursor.execute(f'DROP TABLE IF EXISTS {table_name};')
    # Also drop the view if it exists
    view_name = table_name.replace("Spans", "Traces")
    cursor.execute(f'DROP VIEW IF EXISTS {view_name};')
    cmd = f'''CREATE TABLE {table_name} (
        traceId VARCHAR(50),
        spanId VARCHAR(50),
        parentId VARCHAR(50),
        startTime INTEGER,
        endTime INTEGER,
        serviceName VARCHAR(100),
        status BOOLEAN,
        str_feature_1 VARCHAR(255),
        str_feature_2 VARCHAR(255),
        int_feature_1 INTEGER,
        int_feature_2 INTEGER,
        int_feature_3 INTEGER
    );'''
    cursor.execute(cmd)
    view_name = table_name.replace("Spans", "Traces")
    cursor.execute(f'''
    CREATE VIEW {view_name} AS
        SELECT DISTINCT traceId
        FROM {table_name};
    ''')
    conn.commit()


def fill_data(conn: sqlite3.Connection, traces_dir: str, table_name: str, start_tx: int = 0, end_tx: int = -1):
    try:
        create_spans_table(conn, table_name)
    except Exception as e:
        print("failed to create table", table_name, "error:", e)
    cursor = conn.cursor()
    cursor.execute(f'DELETE FROM {table_name};')
    for tx in get_all_txs(start_tx, end_tx, traces_dir):
        tx_id = tx["details"]["transactionId"]
        child_to_parent = {n["target"]: n["source"] for n in tx["graph"]["edges"]}
        for node_id, node in tx["nodesData"].items():
            features = node.get('environmentVariables', {}).get('body', {})
            if len(features) == 5:
                int_features = [(None, None, int(v)) for v in features.values() if isinstance(v, int)]
                string_features = [(None, None, str(v)) for v in features.values() if isinstance(v, str)]
            else:
                int_features, string_features = extract_metadata(node)
            string_features = ([feature[2] for feature in string_features] + [""] * 2)[:2]
            int_features = ([feature[2] for feature in int_features] + [0] * 3)[:3]
            start_time = node["startTime"]
            end_time = node["startTime"] + node["duration"]
            component_name = node["gent_name"].split('*')[0]
            has_error = 1 if node["issues"] else 0
            cursor.execute(f'''INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                tx_id,
                node_id + tx_id,
                (child_to_parent.get(node_id) or "top") + tx_id,
                start_time,
                end_time,
                component_name,
                has_error,
                *string_features,
                *int_features,
            ))
    conn.commit()

def evaluate_ctgan_dim(conn: sqlite3.Connection, results_dir: str):
    syn_tables = []
    for dim in [(128,), (128,128), (256,), (256,256)]:
        dim = "_".join(map(str, dim))
        syn_tables.append(f"SynSpansCTGANDim{dim}")
        fill_data(
            conn,
            f"{os.path.join(results_dir, dim, "normalized_data")}",
            syn_tables[-1]
        )
    results = {}
    # monitor_errors(syn_tables)
    results["trigger_correlation"] = trigger_correlation(conn, syn_tables)
    results["relative_duration"] = relative_duration(conn, syn_tables, groups=['s1', 's2', 'timeBucket'])
    #attributes(syn_tables, attr_name='str_feature_2', with_sampling=True)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ctgan dim experiment")
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to store generated gent traces')
    parser.add_argument('--db_input', type=str, default='baseline.db', help='Input database file')
    parser.add_argument('--db_output', type=str, default='baseline_and_gent.db', help='Output database file')
    parser.add_argument('--evaluation_results', type=str, default='evaluation_results.json', help='Output file for evaluation results') 
    args = parser.parse_args()

    # copy the baseline database to the output database
    shutil.copy(args.db_input, args.db_output)
    results = evaluate_ctgan_dim(sqlite3.connect(args.db_output), args.results_dir)
    json.dump(results, open(args.evaluation_results, 'w'), indent=4)
