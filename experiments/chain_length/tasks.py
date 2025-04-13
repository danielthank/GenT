import re
import pandas
import sqlite3
import numpy as np
from typing import List, Tuple
from constants import SECOND, ALL_SAMPLINGS

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

def trigger_correlation(conn: sqlite3.Connection, syn_tables: List[str], with_sampling: bool = True, seconds: int = 60):
    def build_query(sampling: str, table_name: str) -> str:
        return f'''
SELECT DISTINCT ROUND(S1.startTime / {seconds * SECOND}) as timeBucket, S1.serviceName as S1, S2.serviceName as S2
FROM {table_name} as S1, {table_name} as S2
Where 
    S1.spanId = S2.parentId
    AND S1.traceId in {sampling}
'''
    no_sample = {
        n: pandas.read_sql_query(build_query(*n), conn)
        for n in set(sample_name_by_syn_table(s) for s in syn_tables + ALL_SAMPLINGS)
    }
    raw_triggers = {
        **{
            syn: [pandas.read_sql_query(build_query(sampling=syn.replace("Spans", "Traces"), table_name=syn), conn) for
                  _ in range(5 if 'HeadBased' in syn else 1)]
            for syn in syn_tables
        },
        **({
               s: [pandas.read_sql_query(build_query(sampling=s, table_name='Spans'), conn) for _ in range(5)]
               for s in ALL_SAMPLINGS
           } if not with_sampling else {})
    }
    results = {}

    #print(syn_tables)
    #print(no_sample.keys())
    print(raw_triggers)
    # print("no_sample", next(iter(no_sample.values()))['timeBucket'].nunique())
    # print("raw_triggers", raw_triggers["SynSpansChainLength2"][0]['timeBucket'].nunique())

    to_trigger_set = lambda triggers: set(
        '#'.join(map(str, t)) for t in triggers[['timeBucket', 'S1', 'S2']].values.tolist())

    all_no_sample = {n: to_trigger_set(data) for n, data in no_sample.items()}
    #print(all_no_sample)
    for sampling_method, triggers_rep in raw_triggers.items():
        all_triggers = all_no_sample[sample_name_by_syn_table(sampling_method)]
        f1_rep = []
        for triggers in triggers_rep:
            triggers = to_trigger_set(triggers)
            print("all_triggers")
            print(all_triggers)
            print("triggers")
            print(triggers)
            tp = len(all_triggers.intersection(triggers))
            fp = len(triggers.difference(all_triggers))
            fn = len(all_triggers.difference(triggers))
            f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0
            print("tp")
            print(tp)
            print("fp")
            print(fp)
            print("fn")
            print(fn)
            f1_rep.append(f1)
        results[sampling_method] = np.average(f1_rep), np.std(f1_rep)
    print(f"trigger_correlation:", results)
    return results