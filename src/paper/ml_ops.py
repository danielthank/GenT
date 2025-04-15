import contextlib
import multiprocessing
import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Iterable
from drivers.base_driver import BaseDriver
from drivers.gent.data import ALL_TRACES
from drivers.gent.gent_driver import GenTDriver
from ml.app_utils import GenTConfig
from paper.ops_utils import FidelityResult, load_results, store_and_upload_results


def measure_configuration(
    driver: BaseDriver,
    lock: Optional[multiprocessing.synchronize.Lock] = None,
    skip_if_exists: bool = True,
) -> FidelityResult:
    if skip_if_exists and driver.get_results_key() in load_results(driver):
        existing_result = load_results(driver)[driver.get_results_key()]
        print(f"Already processed {driver.get_results_key()}")
        return existing_result

    print("Starting", driver.get_driver_name(), driver.gen_t_config.to_string())

    driver.train_and_generate()
    result = FidelityResult(
        model_size=driver.get_model_size(),
        gzip_model_size=driver.get_model_gzip_size(),
    )
    print(f"Driver: {driver}, Result: {result}")
    with lock or contextlib.suppress():
        store_and_upload_results(driver, result, driver.get_driver_name())
    return result


def measure_multiple_configurations(drivers: Iterable[BaseDriver], max_workers=1) -> None:
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        lock = multiprocessing.Manager().Lock()
        futures = [
            pool.submit(measure_configuration, driver, lock, True) for driver in drivers
        ]
        result = [f.result() for f in futures]
    print(result)

def batch_size(traces_dir: str) -> None:
    print("GenT changing CTGAN's generator dimension")
    measure_multiple_configurations(map(GenTDriver, [
        GenTConfig(chain_length=2, tx_end=10_000, traces_dir=traces_dir),
        GenTConfig(chain_length=2, tx_end=15_000, traces_dir=traces_dir),
        GenTConfig(chain_length=2, tx_end=20_000, traces_dir=traces_dir),
    ]), max_workers=3)


def simple_ablations(traces_dir: str) -> None:
    print("GenT simple_ablations")
    configs = [
        GenTConfig(chain_length=2, tx_start=0, tx_end=ALL_TRACES, iterations=10, independent_chains=True, traces_dir=traces_dir),
        GenTConfig(chain_length=2, tx_start=0, tx_end=ALL_TRACES, iterations=10, with_gcn=False, traces_dir=traces_dir),
        GenTConfig(chain_length=2, tx_start=0, tx_end=ALL_TRACES, iterations=10, start_time_with_metadata=True, traces_dir=traces_dir),
    ]
    for i, config in enumerate(configs):
        print(f"####### simple_ablations index: {i} ########")
        measure_configuration(GenTDriver(config), skip_if_exists=False)


def main():
    parser = argparse.ArgumentParser(description='Run GenT experiments')
    parser.add_argument('experiment', type=str, help='Experiment to run (ctgan_dim, iterations, simple_ablations, batch_size)')
    parser.add_argument('--traces_dir', type=str, required=True, help='Directory containing trace data')
    args = parser.parse_args()
    
    if args.experiment == "simple_ablations":
        simple_ablations(args.traces_dir)
    elif args.experiment == "batch_size":
        batch_size(args.traces_dir)
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")

if __name__ == "__main__":
    main()
    # iterations = 3
    # for desc in os.listdir("/Users/saart/cmu/GenT/traces"):
    #     if desc == "wildryde":
    #         continue
    #     for i in range(iterations):
    #         traces_dir = f"/Users/saart/cmu/GenT/traces/{desc}"
    #         driver = GenTDriver(GenTConfig(chain_length=2, iterations=30, tx_end=10000 + i, traces_dir=traces_dir))
    #         driver.train_and_generate()
    #         fill_benchmark(real_data_dir=traces_dir, syn_data_dir=driver.get_generated_data_folder(), desc=desc, variant=i)
    # driver = TabFormerDriver(GenTBaseConfig(chain_length=3, iterations=5))
    # measure_configuration(driver, skip_if_exists=False)
    # iterations_exp()
    # ctgan_dim()
    # simple_ablations()