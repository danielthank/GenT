import json
import os
import sys
import requests
from typing import Optional, List
from requests.exceptions import RequestException
from ml.app_denormalizer import Component, prepare_tx_structure

import argparse

JAEGER_URL = "http://localhost:16686"


def download_traces_from_jaeger(service_name: str, jaeger_url: str, target_dir: str) -> int:
    response = requests.get(f"{jaeger_url}/api/traces?service={service_name}&limit=10000").json()
    traces = []
    for trace in response["data"]:
        trace_id = trace["traceID"]
        response = requests.get(f"{jaeger_url}/api/traces/{trace_id}").json()
        traces.extend(response["data"])
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, f"{service_name}.json")
    with open(target_file, "w") as f:
        json.dump(traces, f, indent=4)
    # print(f"Downloaded {len(traces)} traces for {service_name} to {target_file}.")
    return len(traces)


def download_traces_from_jaeger_for_all_services(target_dir: str, jaeger_url: str = JAEGER_URL) -> int:
    response = requests.get(f"{jaeger_url}/api/services")
    total = 0
    all_services = response.json()["data"] or []
    for service in all_services:
        if "jaeger" in service:
            # print("Skipping jaeger service", service)
            continue
        total += download_traces_from_jaeger(service, jaeger_url, target_dir)
    return total


def _handle_jaeger_trace(jaeger_trace: dict) -> dict:
    def get_service_name(s):
        for tag in s["tags"]:
            if tag["key"] == "http.url":
                hostname = tag["value"].split('?')[0]
                if hostname.startswith("http://"):
                    hostname = hostname[len("http://"):]
                return hostname
        return jaeger_trace["processes"][s["processID"]]["serviceName"]
    span_to_service_name = {s["spanID"]: (get_service_name(s), s["startTime"]) for s in jaeger_trace["spans"]}
    span_id_to_ts_name = {}
    components = []
    for span in jaeger_trace["spans"]:
        service_name, start_time = span_to_service_name[span["spanID"]]
        dup_names = [t for s, t in span_to_service_name.values() if s == service_name]
        dup_names = sorted(dup_names)
        name = f'{service_name}*{dup_names.index(start_time)}'
        span_id_to_ts_name[span["spanID"]] = name
    for span in jaeger_trace["spans"]:
        components.append(Component(
            component_id=span_id_to_ts_name[span["spanID"]],
            start_time=span["startTime"],
            end_time=span["startTime"] + span["duration"],
            has_error=any(
                ((tag["key"] == "error" and tag["value"] is True)
                or tag["key"] == "http.status_code" and tag["value"] > 300)
                for tag in span["tags"]),
            # TODO: change children_ids to parent_ids
            children_ids=[span_id_to_ts_name.get(ref["spanID"]) for ref in span["references"] if ref["refType"] == "CHILD_OF"],
            group="",
            metadata={t["key"]: t["value"] for t in span["tags"]} | {f"process_{t['key']}": t["value"] for t in jaeger_trace['processes'][span["processID"]]['tags']},
            component_type="jaeger",
            duration=span["duration"]
        ))
    return prepare_tx_structure(transaction_id=jaeger_trace["traceID"], components=components)


def translate_jaeger_to_gent(from_dir: str, to_dir: Optional[str] = None) -> None:
    if not to_dir:
        to_dir = from_dir.replace("raw_jaeger", "gent")
    os.makedirs(to_dir, exist_ok=True)
    for service_file in os.listdir(from_dir):
        with open(os.path.join(from_dir, service_file)) as f:
            jaeger_traces = json.load(f)
        translate_jaeger_to_gent_from_list(jaeger_traces, os.path.join(to_dir, service_file))


def translate_jaeger_to_gent_from_list(jaeger_traces: List[dict], filepath: Optional[str] = None) -> None:
        root_cause_filepath = filepath.rsplit('.', 1)[0] + '.root_cause.json'
        
        success = 0
        fail = 0
        with open(filepath, "w") as f, open(root_cause_filepath, "w") as rf:
            for jaeger_trace in jaeger_traces:
                gent_trace = _handle_jaeger_trace(jaeger_trace)
                if gent_trace:
                    success += 1
                    f.write(json.dumps(gent_trace) + ",\n")
                    if "rootCause" in jaeger_trace:
                        root_cause_data = {
                            "traceId": jaeger_trace["traceID"],
                            "rootCause": jaeger_trace["rootCause"]
                        }
                        rf.write(json.dumps(root_cause_data) + ",\n")
                else:
                    fail += 1
        print(f"Translated {success} traces, {fail} failed.")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Jaeger traces to GenT format with options to download or translate existing traces.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='action', required=True, help='Action to perform')

    # Translate subcommand
    translate_parser = subparsers.add_parser('translate', 
        help='Translate existing Jaeger traces to GenT format')
    translate_parser.add_argument('--from-dir', required=True,
        help='Directory containing Jaeger traces')
    translate_parser.add_argument('--to-dir',
        help='Output directory for translated traces (default: replaces raw_jaeger with gent in from-dir)')

    # Download-and-translate subcommand
    download_parser = subparsers.add_parser('download-and-translate',
        help='Download traces from Jaeger and translate them to GenT format')
    download_parser.add_argument('--target-dir', required=True,
        help='Directory to store downloaded traces')
    download_parser.add_argument('--app', required=True,
        help='Application name')
    download_parser.add_argument('--jaeger-url', default=JAEGER_URL,
        help=f'Jaeger API endpoint (default: {JAEGER_URL})')
    download_parser.add_argument('--to-dir',
        help='Output directory for translated traces (default: replaces raw_jaeger with gent in target-dir)')

    args = parser.parse_args()

    try:
        if args.action == 'translate':
            translate_jaeger_to_gent(
                from_dir=args.from_dir,
                to_dir=args.to_dir
            )
        else:  # download-and-translate
            target_dir = os.path.join(args.target_dir, "raw_jaeger")
            download_traces_from_jaeger_for_all_services(
                target_dir=target_dir,
                jaeger_url=args.jaeger_url
            )
            to_dir = args.to_dir if args.to_dir else os.path.join(args.target_dir, "gent")
            translate_jaeger_to_gent(
                from_dir=target_dir,
                to_dir=to_dir
            )
    except RequestException as e:
        print(f"Error accessing Jaeger API: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
