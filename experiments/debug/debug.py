import os
import argparse
import pickle
from joblib import parallel_backend
from ml.app_utils import GenTConfig
from drivers.gent.start_time_generator_ctgan import StartTimesGenerator
from drivers.gent.metadata_generator_ctgan import MetadataGenerator

ALL_TRACES = 9342

def test_start_time(config):
    # driver = GenTDriver(config)

    start_time_generator = StartTimesGenerator.get(config)
    start_time_generator.train()
    timestamps = start_time_generator.generate_timestamps_corpus()
    print(f"Generated {len(timestamps)} timestamps")
    print(timestamps)

    os.makedirs(config.results_dir, exist_ok=True)
    path = os.path.join(config.results_dir, "timestamps.pkl")
    pickle.dump(timestamps, open(path, "wb"))

    start_time_generator.save()


def test_metadata(config):
    # driver = GenTDriver(config)

    path = os.path.join(config.results_dir, "timestamps.pkl")
    timestamps = pickle.load(open(path, "rb"))

    metadata_generator = MetadataGenerator.get(config)
    metadata_generator.train_root()
    metadata_generator.train_chained()
    metadata_generator.generate_traces_corpus(config.results_dir, timestamps)
    metadata_generator.save()
    #metadata_generator.compare()

if __name__ == "__main__":
    parallel_backend("threading")
    parser = argparse.ArgumentParser(description="GenT Driver Test")
    parser.add_argument('--traces_dir', type=str, required=True, help='Directory containing trace data')
    parser.add_argument('--models_dir', type=str, required=True, help='Directory to store models')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to store results')
    parser.add_argument('--test', type=str, nargs='+', required=True, help='Tests to run (start_time, metadata)')
    args = parser.parse_args()

    config = GenTConfig(chain_length=3, tx_start=0, tx_end=ALL_TRACES, iterations=10, traces_dir=args.traces_dir, models_dir=args.models_dir, results_dir=args.results_dir)
    if "start_time" in args.test:
        test_start_time(config)
    if "metadata" in args.test:
        test_metadata(config)