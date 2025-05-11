"""
Read files from S3 and then use GenT
"""
import csv
import json
import os
from datetime import datetime
from typing import List, Iterator

import boto3
import gzip

from dataclasses import dataclass, field

from ml.app_normalizer import get_csv_headers, extract_rows_from_transaction
from ml.app_utils import GenTConfig


@dataclass
class Component:
    transaction_id: str
    name: str
    start_time: datetime
    end_time: datetime
    has_error: bool
    children: List["Component"] = field(default_factory=list)

    def print(self, indent: int = 0):
        print(" " * indent, self.name, self.start_time, self.end_time, self.has_error)
        for child in self.children:
            child.print(indent + 2)
            print("---")

    def get_nodes(self) -> List["Component"]:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_nodes())
        return nodes

    def get_app_edges(self):
        edges = []
        for child in self.children:
            edges.append({"source": self.name, "target": child.name})
            edges.extend(child.get_app_edges())
        return edges

    def to_app_transaction(self) -> dict:
        return {
            "details": {
                "transactionId": self.transaction_id,
                "startTime": self.start_time.timestamp() * 1000,
            },
            "graph": {
                "edges": self.get_app_edges()
            },
            "nodesData": {
                node.name: {
                    "id": node.name,
                    "resource": {"name": node.name, "serviceType": "otel"},
                    "duration": (self.end_time - self.start_time).total_seconds() * 1000,
                    "startTime": self.start_time.timestamp() * 1000,
                    "issues": ["error"] if node.has_error else [],
                }
                for node in self.get_nodes()
            }
        }


def read_raw_telemetries(bucket_name: str):
    bucket = boto3.resource('s3').Bucket(bucket_name)
    for obj in bucket.objects.all():
        data = gzip.decompress(obj.get()['Body'].read()).decode()
        for record_str in data.split('{"messageType"')[1:]:
            record = json.loads('{"messageType"' + record_str)
            if record['messageType'] == 'DATA_MESSAGE':
                yield json.loads(record['logEvents'][0]['message'])["msg"]


def build_transactions(spans: List[dict], parent_id: str = "") -> List[Component]:
    children = []
    for node in spans:
        if node['Parent ID'] == parent_id:
            if "faas.id" in node:
                name = node["faas.id"].split(":")[-1].strip(")")
            else:
                name = node["http.url"][11:-1]
            children.append(Component(
                transaction_id=node["Trace ID"],
                name=name,
                start_time=datetime.strptime(node["Start time"].split("+")[0][:-5], "%Y-%m-%d %H:%M:%S.%f"),
                end_time=datetime.strptime(node["End time"].split("+")[0][:-5], "%Y-%m-%d %H:%M:%S.%f"),
                has_error=node["Status code"] == 'Error',
                children=[c for c in build_transactions(spans, node["ID"]) if c is not None],
            ))
    return children


def parse_raw_telemetry_to_dict(raw_telemetry: str) -> List[dict]:
    spans: List[dict] = []
    current_span = None
    for line in raw_telemetry.split('\n'):
        if line.startswith('Span #'):
            if current_span:
                spans.append(current_span)
            current_span = {}
        if (current_span is not None) and (':' in line):
            key, *value = line.split(': ', 1)
            current_span[key.strip(' ->')] = ":".join(value).strip()
    spans.append(current_span)
    return spans


def get_transactions(bucket_name: str) -> Iterator[Component]:
    spans: List[dict] = []
    for raw_telemetry in read_raw_telemetries(bucket_name=bucket_name):
        spans.extend(parse_raw_telemetry_to_dict(raw_telemetry))
    return build_transactions(spans)


def normalize_data(bucket_name: str, config: GenTConfig) -> None:
    out_dir = config.get_raw_normalized_data_dir() + "collector"
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "output.csv")
    with open(output_path, "w") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(get_csv_headers(config))
        for component in get_transactions(bucket_name):
            transaction = component.to_app_transaction()
            for row in extract_rows_from_transaction(transaction, config):
                writer.writerow(row)



if __name__ == '__main__':
    normalize_data(bucket_name="gent", config=GenTConfig())
