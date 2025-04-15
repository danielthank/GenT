import argparse
import os
from drivers.base_driver import BaseDriver
from drivers.gent.gent_driver import GenTDriver
from ml.app_utils import GenTConfig
from paper.ops_utils import FidelityResult, load_results, store_results

ALL_TRACES = 9342

def measure_configuration(
    driver: BaseDriver,
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
    store_results(driver, result, driver.get_driver_name())
    return result

def tx_iteration(traces_dir: str, models_dir: str, results_dir: str) -> None:
    configs = [
        GenTConfig(
            chain_length=2,
            tx_start=0,
            tx_end=tx_count,
            iterations=iterations,
            traces_dir=traces_dir,
            models_dir=os.path.join(models_dir, f"{tx_count}_{iterations}"),
            results_dir=os.path.join(results_dir, f"{tx_count}_{iterations}"),
        )
        for tx_count in [1_000, 2_000, 5_000, 10_000]
        for iterations in [1, 2, 3, 4, 5, 6, 7, 10, 20, 30]
    ]
    for config in configs:
        print(f"#### iterations {config.iterations} tx_count {config.tx_end} #####")
        measure_configuration(GenTDriver(config), skip_if_exists=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TX and Iteration Experiment")
    parser.add_argument('--traces_dir', type=str, required=True, help='Directory containing trace data')
    parser.add_argument('--models_dir', type=str, required=True, help='Directory to store models')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to store results')
    args = parser.parse_args()

    tx_iteration(args.traces_dir, args.models_dir, args.results_dir)
