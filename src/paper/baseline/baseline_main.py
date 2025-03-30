import os
import shutil
import subprocess
import argparse
from drivers.tabFormer.tab_former_driver import TAB_FORMER_INTERPRETER, BASE_GENERATE_PARAMS, BASE_TRAIN_PARAMS, \
    TAB_FORMER_DIR, TabFormerDriver
from drivers.netshare.netshare_driver import NetShareDriver
from fidelity.bottlenecks import get_bottleneck_score
from fidelity.compare_forests import get_forest_score
from fidelity.monitor import get_monitor_score
from paper.baseline.app_baseline_denormalizer import denormalize_data_baseline
from paper.baseline.app_baseline_normalizer import normalize_data_baseline
from paper.baseline.baseline_utils import store_result

try:
    from netshare.configs import load_from_file
    from netshare.generate.generate import generate
    from netshare.api import Generator
except ImportError:
    pass


TARGET_DIR = os.path.join(os.path.dirname(__file__), "raw_normalized_data")
WORK_FOLDER = os.path.join(os.path.dirname(__file__), "work_folder")

def netshare() -> str:
    generator = Generator(config="./baseline_config.json")
    shutil.rmtree(f"{WORK_FOLDER}/netshare/", ignore_errors=True)
    generator.train()
    NetShareDriver.generate_loop(
        generator=generator,
        generated_data_dir_name=f"{WORK_FOLDER}/netshare/output_data",
        target_rows=120_000 #sum(1 for file in os.listdir(TARGET_DIR) for _ in open(os.path.join(TARGET_DIR, file))),
    )
    return f"{WORK_FOLDER}/netshare/output_data"


def tabformer() -> str:
    shutil.rmtree(f"{WORK_FOLDER}/tabformer/", ignore_errors=True)
    subprocess.check_output([
        TAB_FORMER_INTERPRETER,
        "main.py",
        *BASE_TRAIN_PARAMS,
        "--num_train_epochs",
        "10",
        "--save_steps",
        "100",
        "--data_root",
        TARGET_DIR,
        "--output_dir",
        f"{WORK_FOLDER}/tabformer/",
        "--data_type",
        "observability-baseline",
    ], cwd=TAB_FORMER_DIR)

    max_checkpoint = TabFormerDriver.get_highest_checkpoint(f"{WORK_FOLDER}/tabformer/")
    subprocess.check_output([
            TAB_FORMER_INTERPRETER,
            "gpt_eval.py",
            *BASE_GENERATE_PARAMS,
            "--checkpoint",
            str(max_checkpoint),
            "--data_dir",
            TARGET_DIR,
            "--output_dir",
            f"{WORK_FOLDER}/tabformer/",
            "--data_type",
            "observability-baseline",
            "--lines_to_generate",
            str(sum(1 for file in os.listdir(TARGET_DIR) for _ in open(os.path.join(TARGET_DIR, file))))
        ], cwd=TAB_FORMER_DIR
    )

    return f"{WORK_FOLDER}/tabformer/gpt2-userid-0_nbins-10_hsz-1020/checkpoint-{max_checkpoint}-eval"

def get_score(driver: str) -> None:
    final_data_dir = f"{WORK_FOLDER}/{driver}/final_data"
    monitor_score = get_monitor_score(final_data_dir)
    store_result(f"{driver}_monitor_score", monitor_score)
    print("Monitor score:", monitor_score)
    forest_score = get_forest_score(final_data_dir)
    store_result(f"{driver}_forest_score", forest_score)
    print("Forest score:", forest_score)
    bottleneck_score = get_bottleneck_score(final_data_dir)
    store_result(f"{driver}_bottleneck_score", bottleneck_score)
    print("Bottleneck score:", bottleneck_score)


def main(driver: str, traces_dir: str) -> None:
    normalize_data_baseline(traces_dir, TARGET_DIR)
    output_dir = netshare() if driver == "netshare" else tabformer()
    denormalize_data_baseline(output_dir, f"{WORK_FOLDER}/{driver}/final_data")

    get_score(driver)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process trace data and run analysis')
    parser.add_argument('traces_dir', type=str, help='Directory containing trace data')
    args = parser.parse_args()
    # main("netshare", args.traces_dir)
    # main("tabformer", args.traces_dir)
    get_score("netshare")
    get_score("tabformer")
