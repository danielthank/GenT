import re
import os
import time
import gzip
import json
import pprint
import logging
import sqlite3
import multiprocessing
import argparse
from pathlib import Path
from functools import partial
from collections import Counter, OrderedDict
from typing import List, Dict, Tuple

import numpy
import scipy
import pandas
from matplotlib import pyplot
from ml.app_normalizer import extract_metadata
from paper.adaption_experiment import NON_ROLLING_PATH, ROLLING_PATH
from drivers.gent.data import get_all_txs, ALL_TRACES, ROLLING_EXPERIMENT_CONFIGS, ROLLING_EXPERIMENT_NAMES
from pandora_trace.query_db import BENCHMARK_QUERIES, run_templates

FEATURES = ["str_feature_1", "str_feature_2", "int_feature_1", "int_feature_2", "int_feature_3"]
HOUR = 60 * 60 * 1000
SECOND = 1000 * 1000

SAMPLE_SIZES = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150]

HEAD_SAMPLINGS: List[str] = [f"HeadBasedTraces{s}" for s in SAMPLE_SIZES]
ALL_SAMPLINGS: List[str] = HEAD_SAMPLINGS + ["ErrorBasedTraces", "DurationBasedTraces"]
pandas.set_option('display.max_rows', 100)
pandas.set_option('display.max_columns', 10)

conn = sqlite3.connect('./paper/spans_deathstar.db')


def init(traces_dir: str):
    create_spans_table("Spans")
    create_spans_table("SynSpans")
    create_sampling_views()
    fill_data(traces_dir, "Spans")


def create_spans_table(table_name: str):
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


def fill_data(traces_dir: str, table_name: str, start_tx: int = 0, end_tx: int = -1):
    try:
        create_spans_table(table_name)
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


def create_sampling_views(table_prefix: str = ""):
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


def spans_count():
    print(pandas.read_sql_query("SELECT COUNT(*) FROM spans", conn))


def all_services(table_name: str = "", table_prefix: str = "") -> List[str]:
    table_name = table_name or f"{table_prefix}Spans"
    return pandas.read_sql_query(f"SELECT DISTINCT serviceName FROM {table_name}", conn)['serviceName'].values.tolist()


def all_entry_points(table_prefix: str = "") -> List[str]:
    return \
        pandas.read_sql_query(f"SELECT DISTINCT serviceName FROM {table_prefix}Spans WHERE parentId like 'top%'", conn)[
            'serviceName'].values.tolist()


def get_table_prefix(table_name) -> str:
    table_prefix = ""
    if "DeathStar" in table_name:
        table_prefix = "DeathStar"
    return table_prefix


def sample_name_by_syn_table(syn_table: str) -> Tuple[str, str]:
    """
    Return the sampling method and the table name
    """
    table_prefix = get_table_prefix(syn_table)
    if f"Rolling{table_prefix}Spans" in syn_table:
        is_sample = re.match(rf"Rolling{table_prefix}Spans(\d+)HeadBased(\d+)", syn_table)
        if is_sample:
            batch, ratio = is_sample.groups()
            return f"RollingTraces{batch}", f"Rolling{table_prefix}Spans{batch}"
        return syn_table.replace(f"Rolling{table_prefix}Spans", f"Rolling{table_prefix}Traces").replace("Syn",
                                                                                                        ""), syn_table.replace(
            "Syn", "")
    if "TxCount" not in syn_table:
        return f"NoSampling{table_prefix}Traces", f"{table_prefix}Spans"
    k_traces = int(re.findall(r"TxCount(\d+)", syn_table)[0]) // 1000
    return f"First{k_traces}KTraces", f"{table_prefix}Spans"


def _monitor_errors(
        service_name: str = 'wildrydes-prod-unicornDispatched', sampling: str = 'TailBasedTraces',
        table_name: str = 'Spans'
) -> Dict[Tuple[float, str], List[float]]:
    query = f'''
SELECT ROUND(S2.startTime / {HOUR}) as timeBucket, S2.serviceName, count(*) as c
FROM {table_name} as S1, {table_name} as S2
Where 
    S1.serviceName = '{service_name}' AND 
    S1.traceId = S2.traceId
    AND S2.status = 1
    AND S1.traceId in {sampling}
GROUP BY timeBucket, S2.serviceName
'''
    res = pandas.read_sql_query(query, conn)

    # Scale up the sampling
    for sample_size in SAMPLE_SIZES:
        if sampling == f'HeadBasedTraces{sample_size}':
            res['c'] = res['c'] * sample_size
            break

    return {(t, n): v['c'].tolist() for (t, n), v in res.groupby(['timeBucket', 'serviceName'])}


def _measure_by_service_monitor(service_name: str, syn_tables: List[str], with_sampling: bool = True) -> Dict[
    str, List[Counter[str]]]:
    def get_res(no_sample_value, sample_value, threshold: int = 10) -> str:
        sample_value = (sample_value or [0])[0]
        no_sample_value = no_sample_value[0]
        if no_sample_value > threshold:
            if sample_value > threshold:
                return "TP"
            else:
                return "FN"
        else:
            if sample_value > threshold:
                return "FP"
            else:
                return "TN"

    no_sample = {
        n: _monitor_errors(service_name, *n)
        for n in set(sample_name_by_syn_table(s) for s in syn_tables + ALL_SAMPLINGS)
    }
    sampling: Dict[str, List[Counter[str]]] = {"NoSamplingTraces": []}
    for syn in syn_tables:
        data = _monitor_errors(service_name, sampling=syn.replace("Spans", "Traces"), table_name=syn)
        for threshold in [10]:
            curr_no_sample = no_sample[sample_name_by_syn_table(syn)]
            sampling[f"{syn}-{threshold}"] = [
                Counter([get_res(curr_no_sample[n], data.get(n), threshold=threshold) for n in curr_no_sample])]

    for curr_sampling in (ALL_SAMPLINGS if with_sampling else []):
        sampling[curr_sampling] = []
        for _ in range(5):
            try:
                curr_no_sample = no_sample[sample_name_by_syn_table(curr_sampling)]
                head = _monitor_errors(service_name, curr_sampling,
                                       table_name=f'{get_table_prefix(curr_sampling)}Spans')
                sampling[curr_sampling].append(
                    Counter([get_res(curr_no_sample[n], head.get(n)) for n in curr_no_sample])
                )
            except Exception as e:
                print("failed to compute head score", service_name, "error:", repr(e))
                raise e

    print(f"finished {service_name}")
    return sampling


def monitor_errors(syn_tables: List[str], with_sampling: bool = True):
    counters_per_service: List[Dict[str, List[Counter[str]]]]
    print("First 4 services")
    with multiprocessing.Pool(4) as p:
        func = partial(_measure_by_service_monitor, syn_tables=syn_tables, with_sampling=with_sampling)
        # counters_per_service = p.map(func, all_services(syn_tables[0])[:4])
        counters_per_service = [func(s) for s in all_services(syn_tables[0])[:4]]

    results = {}
    for sampling_method in counters_per_service[0]:
        f1_scores = [
                        2 * v['TP'] / (2 * v['TP'] + v['FP'] + v['FN']) if v['TP'] + v['FP'] + v['FN'] > 0 else 0
                        for counters in counters_per_service for v in counters[sampling_method]
                    ] or [0]
        tp = [v['TP'] for counters in counters_per_service for v in counters[sampling_method]] or [0]
        fp = [v['FP'] for counters in counters_per_service for v in counters[sampling_method]] or [0]
        fn = [v['FN'] for counters in counters_per_service for v in counters[sampling_method]] or [0]
        tn = [v['TN'] for counters in counters_per_service for v in counters[sampling_method]] or [0]
        try:
            res = {
                "TP": (f'{numpy.average(tp):.3f}'.rjust(7), f'{numpy.std(tp):.2f}'.rjust(7)),
                "FP": (f'{numpy.average(fp):.3f}'.rjust(7), f'{numpy.std(fp):.2f}'.rjust(7)),
                "FN": (f'{numpy.average(fn):.3f}'.rjust(7), f'{numpy.std(fn):.2f}'.rjust(7)),
                "TN": (f'{numpy.average(tn):.3f}'.rjust(7), f'{numpy.std(tn):.2f}'.rjust(7)),
                "F1": (f'{numpy.average(f1_scores):.3f}'.rjust(7), f'{numpy.std(f1_scores):.2f}'.rjust(7)),
                "FPR": (f'{numpy.average(fp) / (numpy.average(fp) + numpy.average(tn)):.3f}'.rjust(7)),
                "FNR": (f'{numpy.average(fn) / (numpy.average(fn) + numpy.average(tp)):.3f}'.rjust(7)),
            }
            print(sampling_method.ljust(36), res)
            results[sampling_method] = numpy.average(f1_scores), numpy.std(f1_scores)
        except Exception as e:
            print("failed to compute scores", sampling_method, "error:", repr(e))
    results.pop("NoSamplingTraces", None)
    print("monitor_errors", results)


def bottlenecks(syn_tables: List[str]):
    all_results = {}
    for hours_count in [4, 12, 24]:
        if hours_count == 4:
            groups = ['s1', 's2', 'timeBucket']
        elif hours_count == 12:
            groups = ['s1', 'timeBucket']
        else:
            groups = ['timeBucket']
        all_results[f"hours_{hours_count}"] = bottlenecks_by_time_range(syn_tables, hours_count, groups=groups)
    print("all bottlenecks results", all_results)


def attributes(syn_tables: List[str], attr_name: str, with_sampling: bool = True):
    def build_query(sampling: str, table_name: str) -> str:
        return f'''
    SELECT S1.serviceName as s1, S1.{attr_name} as attr, count(distinct S1.traceId) as c
    FROM {table_name} as S1, {table_name} as S2
    Where 
        S1.serviceName in (
            SELECT DISTINCT serviceName
            FROM {table_name}
            WHERE parentId like 'top%'
        )
        AND S1.traceId = S2.traceId
        AND S2.status = 0
        AND S1.traceId in {sampling}
    GROUP BY s1, attr
    '''

    no_sample = {
        n: pandas.read_sql_query(build_query(*n), conn)
        for n in set(sample_name_by_syn_table(s) for s in syn_tables + ALL_SAMPLINGS)
    }
    raw_results = {
        **{
            syn: pandas.read_sql_query(build_query(sampling=syn.replace("Spans", "Traces"), table_name=syn), conn)
            for syn in syn_tables
        },
        **{
            s: pandas.read_sql_query(build_query(sampling=s, table_name='Spans'), conn)
            for s in (ALL_SAMPLINGS if with_sampling else [])
        }
    }

    results = {}
    distributions = {
        s: {service_name: Counter({row['attr']: row['c'] for row in data.iloc}) for service_name, data in
            raw.groupby('s1')}
        for s, raw in raw_results.items()
    }
    no_sampling_distributions = {
        s: {service_name: Counter({row['attr']: row['c'] for row in data.iloc}) for service_name, data in
            raw.groupby('s1')}
        for s, raw in no_sample.items()
    }
    if with_sampling:
        distributions["diversityBased"] = {}
        for service_name in no_sampling_distributions[('NoSamplingTraces', 'Spans')]:
            distributions["diversityBased"][service_name] = Counter(
                {k: 1 for k, v in no_sampling_distributions[('NoSamplingTraces', 'Spans')][service_name].items()})
        for s in HEAD_SAMPLINGS:
            ratio = int(re.match(r"HeadBasedTraces(\d+)", s).group(1))
            for service_name in distributions[s]:
                distributions[s][service_name] = Counter(
                    {k: v * ratio for k, v in distributions[s][service_name].items()})
    for method, ratios in distributions.items():
        raw_ratios = no_sampling_distributions[sample_name_by_syn_table(method)]
        f1_scores = []
        for service_name, counter_1 in raw_ratios.items():
            if attr_name.startswith('str'):
                if service_name not in ratios:
                    f1_scores.append(1)
                    continue
                counter_2 = ratios[service_name]
                tp = sum((counter_2 & counter_1).values())
                fp = sum((counter_1 - counter_2).values())
                fn = sum((counter_2 - counter_1).values())
                f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0
                f1_scores.append(f1)
            else:
                if service_name not in ratios:
                    f1_scores.append(0)
                    continue
                distribution_1 = [k for k, v in counter_1.items() for _ in range(v)]
                distribution_2 = [k for k, v in ratios[service_name].items() for _ in range(v)]
                f1_scores.append(scipy.stats.wasserstein_distance(distribution_1, distribution_2))
        results[method] = numpy.average(f1_scores), numpy.std(f1_scores)
    print("attributes", results)
    return results

def trigger_correlation_many(syn_tables: List[str], with_sampling: bool = True):
    all_results = []
    for hour in [1, 2, 3, 4, 5]:
        all_results.append(trigger_correlation(syn_tables, with_sampling=with_sampling, hours=hour))
    final = {
        method: (numpy.average([r[method][0] for r in all_results]), numpy.std([r[method][0] for r in all_results]))
        for method in all_results[0]
    }
    print("trigger_correlation_many", final)


def ctgan_gen_dim():
    print("dimension_experiment")
    syn_tables = []
    folder_names = {
        (128,): r"/home/danny/Repos/Research/GenTAll/traces/socialNetwork/baseline/gent/results/genT/chain_length=2.iterations=10.traces_dir=gent.with_gcn=True.discriminator_dim=(128,).generator_dim=(128,).start_time_with_metadata=False.independent_chains=False.tx_start=0.tx_end=13911",
        (128, 128): r"/home/danny/Repos/Research/GenTAll/traces/socialNetwork/baseline/gent/results/genT/chain_length=2.iterations=10.traces_dir=gent.with_gcn=True.discriminator_dim=(128,).generator_dim=(128, 128).start_time_with_metadata=False.independent_chains=False.tx_start=0.tx_end=13911",
        (256, 256): r"/home/danny/Repos/Research/GenTAll/traces/socialNetwork/baseline/gent/results/genT/chain_length=2.iterations=10.traces_dir=gent.with_gcn=True.discriminator_dim=(128,).generator_dim=(256, 256).start_time_with_metadata=False.independent_chains=False.tx_start=0.tx_end=13911",
        (256,): r"/home/danny/Repos/Research/GenTAll/traces/socialNetwork/baseline/gent/results/genT/chain_length=2.iterations=10.traces_dir=gent.with_gcn=True.discriminator_dim=(128,).generator_dim=(256,).start_time_with_metadata=False.independent_chains=False.tx_start=0.tx_end=13911",
    }
    for gen_dim, folder_name in folder_names.items():
        dim = '_'.join(map(str, gen_dim))
        syn_tables.append(f"SynSpansCTGANDim{dim}")
        fill_data(
            fr"{folder_name}/normalized_data/",
            syn_tables[-1]
        )
    # monitor_errors(syn_tables, with_sampling=False)
    trigger_correlation(syn_tables, with_sampling=False)
    bottlenecks_by_time_range(syn_tables, 60, groups=['s1', 'timeBucket'], with_sampling=False)
    attributes(syn_tables, attr_name='str_feature_2', with_sampling=False)


def simple_ablations():
    ablations = {
        "NoGCN": fr"/Users/saart/cmu/GenT/results/genT/chain_length=2.iterations=10.metadata_str_size=2.metadata_int_size=3.batch_size=10.is_test=False.with_gcn=" + "False" + ".discriminator_dim=(128,).generator_dim=(128,).start_time_with_metadata=False.independent_chains=False.tx_start=0.tx_end=23010/normalized_data/",
        "NoConditioning": fr"/Users/saart/cmu/GenT/results/genT/chain_length=2.iterations=10.metadata_str_size=2.metadata_int_size=3.batch_size=10.is_test=False.with_gcn=True.discriminator_dim=(128,).generator_dim=(128,).start_time_with_metadata=False.independent_chains=" + "True" + ".tx_start=0.tx_end=23010/normalized_data/",
        "NotTimeSplit": fr"/Users/saart/cmu/GenT/results/genT/chain_length=2.iterations=10.metadata_str_size=2.metadata_int_size=3.batch_size=10.is_test=False.with_gcn=True.discriminator_dim=(128,).generator_dim=(128,).start_time_with_metadata=" + "True" + ".independent_chains=False.tx_start=0.tx_end=23010/normalized_data/",
    }
    syn_tables = ["SynSpansChainLength2"]
    for name, path in ablations.items():
        syn_tables.append(f"SynSpansAblation{name}")
        fill_data(path, syn_tables[-1])
    trigger_correlation(syn_tables, with_sampling=False)
    monitor_errors(syn_tables, with_sampling=False)
    bottlenecks_by_time_range(syn_tables, 2, groups=['s1', 'timeBucket'], with_sampling=False)
    attributes(syn_tables, attr_name='str_feature_2', with_sampling=False)


def create_rolling_tables(is_rolling: bool, traces_dir: str):
    prefix = '' if is_rolling else 'No'
    for i, config in enumerate(ROLLING_EXPERIMENT_CONFIGS):
        try:
            create_spans_table(f"{prefix}RollingSpans{i}")
        except Exception as e:
            print("failed to create table", repr(e))
        fill_data(traces_dir, f"{prefix}RollingSpans{i}", start_tx=config.tx_start, end_tx=config.tx_end)

    cursor = conn.cursor()
    for i in range(len(ROLLING_EXPERIMENT_CONFIGS)):
        for sampling in SAMPLE_SIZES:
            try:
                cursor.execute(f'DROP VIEW IF EXISTS RollingTraces{i}HeadBased{sampling};')
                cursor.execute(f'DROP VIEW IF EXISTS RollingSpans{i}HeadBased{sampling};')
                cursor.execute(f'''
                    CREATE VIEW RollingTraces{i}HeadBased{sampling} AS
                        SELECT DISTINCT traceId
                        FROM {prefix}RollingSpans{i}
                        GROUP BY traceId
                        ORDER BY RANDOM()
                        LIMIT (SELECT count(DISTINCT traceId) FROM {prefix}RollingSpans{i}) / {sampling};
                    ''')
                cursor.execute(f'''
                    CREATE VIEW RollingSpans{i}HeadBased{sampling} AS
                        SELECT *
                        FROM {prefix}RollingSpans{i};
                    ''')

            except Exception as e:
                print("failed to create view", repr(e))
    conn.commit()

    generated_paths = [(ROLLING_PATH if is_rolling else NON_ROLLING_PATH) / name / "generated" for name in
                       ROLLING_EXPERIMENT_NAMES]
    for i, path in enumerate(generated_paths):
        try:
            create_spans_table(f"Syn{prefix}RollingSpans{i}")
        except Exception as e:
            print("failed to create table", repr(e))
        fill_data(path, f"Syn{prefix}RollingSpans{i}")


def rolling_experiment(is_rolling, traces_dir: str):
    create_rolling_tables(is_rolling=is_rolling, traces_dir=traces_dir)
    syn_tables = (
            [f"Syn{'' if is_rolling else 'No'}RollingSpans{i}" for i in range(len(ROLLING_EXPERIMENT_CONFIGS))]
            + [f"RollingSpans{i}HeadBased{sampling}" for i in range(len(ROLLING_EXPERIMENT_CONFIGS)) for sampling in
               [5, 10, 20, 50, 100]]
    )

    print("Rolling experiment, is_rolling:", is_rolling)
    trigger_correlation_many(syn_tables, with_sampling=False)
    monitor_errors(syn_tables, with_sampling=False)
    bottlenecks_by_time_range(syn_tables, 4, groups=['s1', 'timeBucket'], with_sampling=False)
    attributes(syn_tables, attr_name='str_feature_2', with_sampling=False)


def get_size(repetitions: int = 5):
    # python3 -m pip install --upgrade clp-logging
    from clp_logging.handlers import CLPFileHandler

    def _get_size(sampling: str, table_name: str):
        result = {}
        df = pandas.read_sql_query(f"SELECT * FROM {table_name} WHERE traceId in {sampling}", conn)
        telemetries = df.to_dict('records')
        json.dump(telemetries, open(f"/tmp/{sampling}.json", "w"))
        result["raw"] = os.path.getsize(f"/tmp/{sampling}.json") / 1024 / 1024
        print("raw size:", result["raw"], "MB")

        start = time.time()
        with gzip.open(f"/tmp/{sampling}.gz", "wb") as f:
            f.write(open(f"/tmp/{sampling}.json", "rb").read())
        result["gzip"] = os.path.getsize(f"/tmp/{sampling}.gz") / 1024 / 1024
        result["gzip_time"] = time.time() - start
        print("GZIP size:", result["gzip"], "MB, Total time:", result["gzip_time"], "seconds")

        clp_handler = CLPFileHandler(Path(f"/tmp/{sampling}.clp.zst"), mode='wb')
        logger = logging.getLogger(sampling)
        logger.addHandler(clp_handler)
        logger.setLevel(logging.INFO)
        start = time.time()
        for t in telemetries:
            logger.info(t)
        clp_handler.ostream.flush()
        result["clp"] = os.path.getsize(f"/tmp/{sampling}.clp.zst") / 1024 / 1024
        result["clp_time"] = time.time() - start
        print("CLP size:", result["clp"], "MB, Total time:", result["clp_time"], "seconds")
        return result

    all_results = {
        "all": _get_size("NoSamplingTraces", "Spans"),
    }
    sampling_raw_results = {f"{sampling}-{i}": _get_size(sampling, "Spans")
                            for i in range(repetitions) for sampling in ALL_SAMPLINGS}
    for sampling in ALL_SAMPLINGS:
        all_results[sampling] = {}
        for key in ["raw", "gzip", "clp", "gzip_time", "clp_time"]:
            all_results[sampling][key] = (
                numpy.average([sampling_raw_results[f"{sampling}-{i}"][key] for i in range(repetitions)]),
                numpy.std([sampling_raw_results[f"{sampling}-{i}"][key] for i in range(repetitions)]),
            )
    pprint.pprint(all_results)


def get_wasserstein_distance(syn, real):
    syn["c"] = syn["c"] / syn["c"].sum()
    real["c"] = real["c"] / real["c"].sum()
    all_features = set(syn["f"].values) | set(real["f"].values)
    syn = syn.set_index("f").reindex(all_features).fillna(0)
    real = real.set_index("f").reindex(all_features).fillna(0)
    return scipy.stats.wasserstein_distance(syn["c"], real["c"])


def get_parameters_values(prefix: str) -> OrderedDict[str, List[str]]:
    return OrderedDict(
        entry_point=all_entry_points(table_prefix=prefix),
        service_name=all_services(table_prefix=prefix),
        service_name2=all_services(table_prefix=prefix),
        attr_name=FEATURES,
        int_attr_name=[f for f in FEATURES if f.startswith("int")],
    )


def fill_benchmark(real_data_dir, syn_data_dir: str, desc: str, variant: int):
    desc = str(desc).replace(".", "").replace('-', '_')
    prefix = f"DeathStarExpLambda{desc}_{variant}"
    try:
        create_spans_table(f"{prefix}Spans")
        create_spans_table(f"Syn{prefix}Spans")
        create_sampling_views(prefix)
    except sqlite3.OperationalError:
        pass
    fill_data(real_data_dir, f"{prefix}Spans")
    fill_data(syn_data_dir, f"Syn{prefix}Spans")


def benchmark_by_exp_lambda(iterations: int = 3):
    for desc in os.listdir("/Users/saart/cmu/GenT/traces"):
        for variant in range(iterations):
            desc = str(desc).replace(".", "").replace('-', '_')
            prefix = f"DeathStarExpLambda{desc}_{variant}"
            try:
                run_templates(prefix)
            except Exception as e:
                if "no such table" in repr(e):
                    continue


def benchmark_by_query(iterations: int = 3):
    for exp_lambda in reversed([0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]):
        print(exp_lambda, ": {")
        for query in BENCHMARK_QUERIES:
            print(f'"{query[0]}"', ": {")
            for desc in os.listdir("/Users/saart/cmu/GenT/traces"):
                if not desc.endswith(str(exp_lambda)):
                    continue
                for variant in range(iterations):
                    desc = str(desc).replace(".", "").replace('-', '_')
                    prefix = f"DeathStarExpLambda{desc}_{variant}"
                    try:
                        run_templates(conn, get_parameters_values(prefix), prefix, queries=[query])
                    except Exception as e:
                        if "no such table" in repr(e):
                            continue
                        else:
                            raise e
            print("},\n")
        print('},\n\n')


def sampling_of_benchmark(exp_lambda: float = 0.001):
    for desc in os.listdir("/Users/saart/cmu/GenT/traces"):
        if not desc.endswith(str(exp_lambda)):
            continue
        for variant in range(3):
            desc = str(desc).replace(".", "").replace('-', '_')
            prefix = f"DeathStarExpLambda{desc}_{variant}"
            try:
                create_sampling_views(prefix)
            except Exception as e:
                if "no such table" in repr(e) or "already exists" in repr(e):
                    pass
                else:
                    raise e
            try:
                cursor = conn.cursor()
                for sampling in SAMPLE_SIZES:
                    cursor.execute(f'''
                    CREATE VIEW HeadBased{prefix}Spans{sampling} AS
                        SELECT *
                        FROM {prefix}Spans
                        WHERE traceId in HeadBased{prefix}Traces{sampling};
                    ''')
                conn.commit()
            except Exception as e:
                if "no such table" in repr(e) or "already exists" in repr(e):
                    continue
                else:
                    raise e


def test_benchmark_sampling(exp_lambda: float = 0.001):
    for sampling in [5, 10, 20, 50, 100]:
        print(sampling, ": {")
        for query in BENCHMARK_QUERIES:
            print(f'"{query[0]}"', ": {")
            for desc in os.listdir("/Users/saart/cmu/GenT/traces"):
                if not desc.endswith(str(exp_lambda)):
                    continue
                for variant in range(3):
                    desc = str(desc).replace(".", "").replace('-', '_')
                    non_real = f"HeadBased{{prefix}}Spans{sampling}"
                    prefix = f"DeathStarExpLambda{desc}_{variant}"
                    try:
                        run_templates(conn, get_parameters_values(prefix), prefix, queries=[query], non_real=non_real)
                    except Exception as e:
                        if "no such table" in repr(e):
                            pass
                        else:
                            raise e
            print("},\n")
        print('},\n\n')


def selected_specifics():
    specifics = [
        (
            "Find bottlenecks",
            {'entry_point': 'user-timeline-service', 'service_name': 'user-timeline-service'},
            'DeathStarExpLambdasocialNetwork_memory_stress_01_001_0',
        ),
        (
            'Frequency of an attribute',
            {'attr_name': 'str_feature_1'},
            'DeathStarExpLambdasocialNetwork_memory_stress_01_001_0',
        ),
        (
            'Find bottlenecks',
            {'entry_point': 'user-timeline-service', 'service_name': 'user-timeline-service'},
            'DeathStarExpLambdasocialNetwork_memory_stress_01_01_0',
        ),
    ]
    for index, (query_name, params, prefix) in enumerate(specifics):
        query = next(q for q in BENCHMARK_QUERIES if q[0] == query_name)[1]
        syn = pandas.read_sql_query(query.format(table_name=f"Syn{prefix}Spans", **params), conn)
        real = pandas.read_sql_query(query.format(table_name=f"{prefix}Spans", **params), conn)
        distance = get_wasserstein_distance(syn, real)
        syn = syn[syn["c"] > 0.007]
        real = real[real["c"] > 0.007]
        pyplot.bar(syn["f"], syn["c"], label="Synthetic", color='blue', alpha=0.7)
        pyplot.bar(real["f"], real["c"], label="Real", color='orange', alpha=0.7)
        pyplot.title(f"{prefix.split('_', 1)[1]} - {query_name} - {index} - {distance:.2f}")
        pyplot.xlim(real["f"].min(), real["f"].max())
        pyplot.legend()
        pyplot.show()
        print({
            "real": {a: b for a, b in zip(real["f"], real["c"])},
            "syn": {a: b for a, b in zip(syn["f"], syn["c"])},
        })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process trace data and run analysis')
    parser.add_argument('--traces_dir', type=str, required=True, help='Directory containing trace data')
    args = parser.parse_args()
    
    init(args.traces_dir)
    #monitor_chain_length()
    #ctgan_gen_dim()
    #monitor_with_syn_tables()
    # bottlenecks(["SynSpansChainLength2"])
    # rolling_experiment(is_rolling=True)
    # simple_ablations()
    # fill_data(
    #     fr"/Users/saart/cmu/GenT/results/genT/chain_length=2.iterations=10.metadata_str_size=2.metadata_int_size=3.batch_size=10.is_test=False.with_gcn=True.discriminator_dim=(128,).generator_dim=(128,).start_time_with_metadata=False.independent_chains=False.tx_start=0.tx_end={ALL_TRACES-1}/normalized_data/",
    #     "SynSpansChainLength2"
    # )
    # attributes(["SynSpansChainLength2"], "str_feature_1", with_sampling=True)
    # monitor_errors(["SynSpansChainLength2"])
    # trigger_correlation(["SynSpansChainLength2"])
    # spans_count()
    # get_size()
    # test_benchmark()
    # benchmark_by_exp_lambda()
    # selected_specifics()
    # benchmark_by_query(iterations=1)
    # sampling_of_benchmark()
    # test_benchmark_sampling()
