import argparse
import sqlite3
from drivers.gent.data import get_all_txs
from ml.app_normalizer import extract_metadata
from fidelity.constants import SAMPLE_SIZES

ALL_TRACES = 9342

def init(conn: sqlite3.Connection, traces_dir: str):
    create_spans_table(conn, "Spans")
    create_spans_table(conn, "SynSpans")
    create_sampling_views(conn)
    fill_data(conn, traces_dir, "Spans")

def create_sampling_views(conn: sqlite3.Connection, table_prefix: str = ""):
    cursor = conn.cursor()
    
    # Drop and recreate the head-based sampling views
    for sampling in SAMPLE_SIZES:
        # Drop the view if it exists
        cursor.execute(f'DROP VIEW IF EXISTS HeadBased{table_prefix}Traces{sampling};')
        
        # Create the view
        cursor.execute(f'''
        CREATE VIEW HeadBased{table_prefix}Traces{sampling} AS
            SELECT DISTINCT traceId
            FROM {table_prefix}Spans
            GROUP BY traceId
            ORDER BY RANDOM()
            LIMIT (SELECT count(DISTINCT traceId) FROM {table_prefix}Spans) / {sampling};
        ''')
    
    # Drop and recreate other sampling views
    cursor.execute(f'DROP VIEW IF EXISTS ErrorBased{table_prefix}Traces;')
    cursor.execute(f'''
    CREATE VIEW ErrorBased{table_prefix}Traces AS
        SELECT DISTINCT traceId
        FROM {table_prefix}Spans
        WHERE status = 1;
    ''')
    
    cursor.execute(f'DROP VIEW IF EXISTS DurationBased{table_prefix}Traces;')
    cursor.execute(f'''
    CREATE VIEW DurationBased{table_prefix}Traces AS
        SELECT DISTINCT traceId
        FROM {table_prefix}Spans
        GROUP BY traceId
        HAVING (max(endTime) - min(startTime)) / 1000 > 190;
    ''')
    
    cursor.execute(f'DROP VIEW IF EXISTS NoSampling{table_prefix}Traces;')
    cursor.execute(f'''
    CREATE VIEW NoSampling{table_prefix}Traces AS
        SELECT DISTINCT traceId
        FROM {table_prefix}Spans;
    ''')
    
    for i in [1, 2, 5, 10, 15]:
        cursor.execute(f'DROP VIEW IF EXISTS First{i}K{table_prefix}Traces;')
        cursor.execute(f'''
        CREATE VIEW First{i}K{table_prefix}Traces AS
            SELECT traceId
            FROM {table_prefix}Spans
            GROUP BY traceId
            ORDER BY min(startTime) ASC
            LIMIT {i}000;
        ''')
    conn.commit()

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Chain Length Experiment")
    parser.add_argument('--traces_dir', type=str, required=True, help='Directory containing trace data')
    parser.add_argument('--db_output', type=str, default='baseline.db', help='Output database file')
    args = parser.parse_args()

    init(sqlite3.connect(args.db_output), args.traces_dir)
