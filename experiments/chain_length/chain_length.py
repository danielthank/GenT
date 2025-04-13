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

def chain_length(traces_dir: str, models_dir: str, results_dir: str) -> None:
    print("GenT chain length")
    configs = [
        GenTConfig(
            chain_length=length,
            tx_start=0,
            tx_end=ALL_TRACES,
            iterations=10,
            traces_dir=traces_dir,
            models_dir=os.path.join(models_dir, str(length)),
            results_dir=os.path.join(results_dir, str(length)),
        ) for length in [2, 3, 4, 5]
    ]
    for config in configs:
        print("#### Chain length #####", config.chain_length)
        measure_configuration(GenTDriver(config), skip_if_exists=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chain Length Experiment")
    parser.add_argument('--traces_dir', type=str, required=True, help='Directory containing trace data')
    parser.add_argument('--models_dir', type=str, required=True, help='Directory to store models')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to store results')
    args = parser.parse_args()

    chain_length(args.traces_dir, args.models_dir, args.results_dir)
