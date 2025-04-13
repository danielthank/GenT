import argparse
import pickle
from ml.app_utils import GenTConfig
from drivers.gent.gent_driver import GenTDriver
from drivers.gent.start_time_generator_ctgan import StartTimesGenerator
from drivers.gent.metadata_generator_ctgan import MetadataGenerator

ALL_TRACES = 9342

def test_start_time(config):
    # driver = GenTDriver(config)

    start_time_generator = StartTimesGenerator.get(config)
    start_time_generator.train()
    timestamps_by_graph = start_time_generator.generate_timestamps_corpus()
    print()
    print("-" * 20)
    print(f"Generated {len(timestamps_by_graph)} timestamps")
    for graph, timestamps in timestamps_by_graph.items():
        print(f"Graph {graph}: {len(timestamps)} timestamps")
    # print({k: v[:2] for k, v in timestamp.items()})
    min_timestamp = min([t for v in timestamps_by_graph.values() for t in v])
    max_timestamp = max([t for v in timestamps_by_graph.values() for t in v])
    print(f"Min timestamp: {min_timestamp}")
    print(f"Max timestamp: {max_timestamp}")

    start_time_generator.compare()

    pickle.dump(timestamps_by_graph, open("timestamps_by_graph.pkl", "wb"))

    start_time_generator.save()


def test_metadata(config):
    # driver = GenTDriver(config)

    timestamps_by_graph = pickle.load(open("timestamps_by_graph.pkl", "rb"))

    metadata_generator = MetadataGenerator.get(config)
    metadata_generator.train_root()
    metadata_generator.train_chained()
    metadata_generator.generate_traces_corpus("metadata_result", timestamps_by_graph)
    metadata_generator.save()
    #metadata_generator.generate_metadata_corpus()
    #metadata_generator.compare()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenT Driver Test")
    parser.add_argument('--traces_dir', type=str, required=True, help='Directory containing trace data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output data')
    parser.add_argument('--test', type=str, nargs='+', required=True, help='Tests to run (start_time, metadata)')
    args = parser.parse_args()

    config = GenTConfig(chain_length=3, tx_start=0, tx_end=ALL_TRACES, iterations=10, traces_dir=args.traces_dir, output_dir=args.output_dir)
    if "start_time" in args.test:
        test_start_time(config)
    if "metadata" in args.test:
        test_metadata(config)