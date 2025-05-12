import json
import pickle
import uuid
import os
import pandas as pd

from rdt.transformers import LogScaler
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, List, Tuple, Union, Set
from drivers.gent.data import get_full_dataset_chains
from ml.app_denormalizer import prepare_components, prepare_tx_structure
from ml.app_utils import GenTConfig
from gent_utils.utils import NpEncoder

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# TODO: fix this when adding metadata
def get_node_columns(index: int):
    """
    metadata_columns = [
        'gapFromParent_{i}', 'duration_{i}', 'hasError_{i}', 'metadata_{i}_0', 'metadata_{i}_1',
        'metadata_{i}_2', 'metadata_{i}_3', 'metadata_{i}_4'
    ]
    """
    metadata_columns = [
        'gapFromParent_{i}', 'duration_{i}'
    ]
    return [c.format(i=index) for c in metadata_columns]

class MetadataGenerator:
    def __init__(self, gen_t_config: GenTConfig, is_roll: bool = False) -> None:
        self.gen_t_config = gen_t_config
        self.models_dir = os.path.join(gen_t_config.models_dir, "metadata")
        self.n_epochs = self.gen_t_config.iterations
        self.root_generator: Optional[CTGANSynthesizer] = None
        self.chained_generator: Optional[CTGANSynthesizer] = None
        self.graph_index_to_chains: Dict[int, Tuple[List[int], List[int]]] = {}
        self.all_graph_values: List[str] = []
        self.all_chain_values: List[str] = []
        self.node_to_index = {}
        self.is_roll = is_roll
        assert self.gen_t_config.chain_length >= 2, "second node is conditioned by the first"

    def train(self):
        self.train_root()
        self.train_chained()

    def prepare(self) -> Tuple[pd.DataFrame, pd.Series, List[str], int]:
        dataset, all_dataset = get_full_dataset_chains(self.gen_t_config, load_all=self.is_roll)
        dataset = dataset.copy()

        self.all_graph_values = sorted(all_dataset["graph"].unique())
        self.all_chain_values = sorted(all_dataset["chain"].unique())
        all_node_values = set()
        for graph in self.all_graph_values:
            for a, b in eval(graph):
                all_node_values.add(a)
                all_node_values.add(b)
        all_node_values = sorted(all_node_values) 
        self.node_to_index = {node_value: i for i, node_value in enumerate(all_node_values)}

        self.graph_index_to_chains = {index: (
            all_dataset[(all_dataset["graph"] == graph) & (all_dataset["is_root_chain"] == True)]["chain"].apply(self.all_chain_values.index).unique().tolist(),
            all_dataset[(all_dataset["graph"] == graph) & (all_dataset["is_root_chain"] == False)]["chain"].apply(self.all_chain_values.index).unique().tolist(),
        ) for index, graph in enumerate(self.all_graph_values)}

        dataset["graph"] = dataset["graph"].apply(self.all_graph_values.index)
        dataset["chain"] = dataset["chain"].apply(self.all_chain_values.index)

        dataset.drop(columns="txStartTime", inplace=True)

        return dataset
    
    def _get_sdv_metadata(self):
        metadata = Metadata()
        metadata.add_table("metadata")
        metadata.add_column("graph", sdtype="categorical")
        metadata.add_column("chain", sdtype="categorical")
        for i in range(self.gen_t_config.chain_length):
            metadata.add_column(f"gapFromParent_{i}", sdtype="numerical")
            metadata.add_column(f"duration_{i}", sdtype="numerical")
        metadata.add_column("is_root_chain", sdtype="boolean")
        return metadata
    
    def _get_customized_transformers(self):
        customized_transformer = {}
        for i in range(self.gen_t_config.chain_length):
            customized_transformer[f"gapFromParent_{i}"] = LogScaler(constant=-0.001)
            customized_transformer[f"duration_{i}"] = LogScaler(constant=-0.001)
        return customized_transformer
    
    def _get_synthesizer(self) -> CTGANSynthesizer:
        metadata = self._get_sdv_metadata()
        synthesizer = CTGANSynthesizer(
            metadata=metadata,
            epochs=self.n_epochs,
            batch_size=self.gen_t_config.batch_size,
            generator_dim=self.gen_t_config.generator_dim,
            discriminator_dim=self.gen_t_config.discriminator_dim,
            enforce_rounding=False,
            verbose=True,
        )
        return synthesizer

    def train_root(self):
        print("Metadata root generator started training")
        self.root_generator = self.root_generator or self._get_synthesizer()

        dataset = self.prepare()
        relevant_indexes = (dataset["is_root_chain"] == True)
        root_dataset = dataset[relevant_indexes]
        self.root_generator.auto_assign_transformers(root_dataset)
        self.root_generator.update_transformers(self._get_customized_transformers())
        self.root_generator.fit(root_dataset)

    def train_chained(self):
        print("Metadata chain generator started training")
        self.chained_generator = self.chained_generator or self._get_synthesizer()
        
        dataset = self.prepare()
        relevant_indexes = (dataset["is_root_chain"] == False)
        chained_dataset = dataset[relevant_indexes]
        self.chained_generator.auto_assign_transformers(chained_dataset)
        self.chained_generator.update_transformers(self._get_customized_transformers())
        self.chained_generator.fit(chained_dataset)

    def _save_generator(self, gen: CTGANSynthesizer, name: str):
        path = self.models_dir
        gen.save(f"{path}/{name}_ctgan_generator.pkl")

    def save(self):
        self.save_root()
        self.save_chained()

    def save_root(self):
        path = self.models_dir
        os.makedirs(path, exist_ok=True)
        self._save_generator(self.root_generator, "root")
        pickle.dump(self.all_graph_values, open(f"{path}/all_graph_values.pkl", "wb"))
        pickle.dump(self.all_chain_values, open(f"{path}/all_chain_values.pkl", "wb"))
        pickle.dump(self.node_to_index, open(f"{path}/node_to_index.pkl", "wb"))
        pickle.dump(self.graph_index_to_chains, open(f"{path}/graph_index_to_chains.pkl", "wb"))

    def save_chained(self):
        path = self.models_dir
        os.makedirs(path, exist_ok=True)
        self._save_generator(self.chained_generator, "chained")

    def load(self, only_root: bool = False, only_chained: bool = False):
        path = self.models_dir
        def load_generator(name):
            generator = CTGANSynthesizer.load(f"{path}/{name}_ctgan_generator.pkl")
            return generator

        if not only_chained:
            self.root_generator = load_generator("root")
        if not only_root:
            self.chained_generator = load_generator("chained")

        self.all_graph_values = pickle.load(open(f"{path}/all_graph_values.pkl", "rb"))
        self.all_chain_values = pickle.load(open(f"{path}/all_chain_values.pkl", "rb"))
        self.node_to_index = pickle.load(open(f"{path}/node_to_index.pkl", "rb"))
        self.graph_index_to_chains = pickle.load(open(f"{path}/graph_index_to_chains.pkl", "rb"))

    def _component_data_to_tx(self, graph_index: int, component_data: Dict[str, dict], tx_start_time: int, index_to_node: Dict[int, str]) -> Optional[str]:
        for node in component_data:
            component_data[node]['componentName'] = node
            component_data[node]['txStartTime'] = tx_start_time
            component_data[node]['parentComponentName'] = "top"
            metadata = {}
            for key, value in component_data[node].items():
                # TODO: fix this when adding metadata
                if 'metadata' in key:
                    new_key_name = get_key_name(node, int(key.replace('metadata_', '')),
                                                config=self.gen_t_config)
                    metadata[new_key_name] = get_key_value(node, new_key_name, value, self.gen_t_config)
            component_data[node]['metadata'] = metadata
        graph_str = self.all_graph_values[graph_index]
        for source, target in eval(graph_str):
            if target in component_data:
                component_data[target]['parentComponentName'] = source

        components = prepare_components(None, config=self.gen_t_config,
                                        extracted_component_data=component_data)
        tx = prepare_tx_structure(uuid.uuid1().hex, components)
        return json.dumps(tx, cls=NpEncoder)

    def generate_traces_corpus(
            self, target_dir_path: Union[str, Path], ts_corpus: pd.DataFrame,
    ):
        index_to_node = {index: node for node, index in self.node_to_index.items()}

        chains_to_generate: Dict[Tuple[int, int], Set[int]] = {}
        root_chains_to_generate: Dict[Tuple[int, int], Set[int]] = {}
        generated_graphs: Dict[Tuple[int, int], Dict[str, dict]] = defaultdict(dict)

        for row in ts_corpus.itertuples():
            graph_index = row.graph
            ts = row.startTime
            root_chains, chained_chains = self.graph_index_to_chains[graph_index]
            root_chains_to_generate[(graph_index, ts)] = set(root_chains)
            chains_to_generate[(graph_index, ts)] = set(chained_chains)
            generated_graphs[(graph_index, ts)] = {}

        # TODO: do we need trigger data?
        def generate_bulk(is_root: bool, curr_chain_to_generate: List[Tuple[int, int, int]]):
            known_columns = pd.DataFrame([{
                'graph': graph_index,
                'chain': chain_index,
                'is_root_chain': is_root,
            } for ts, graph_index, chain_index in curr_chain_to_generate])
            bulk_data = (self.root_generator if is_root else self.chained_generator).sample_remaining_columns(
                known_columns=known_columns,
                max_tries_per_batch=500,
            )

            if len(bulk_data) != len(curr_chain_to_generate):
                raise Exception(f"Expected {len(curr_chain_to_generate)} samples but got {len(bulk_data)}")
            for row, (ts, graph_index, chain_index) in zip(iter(bulk_data.iloc), curr_chain_to_generate):
                graph_nodes = self.all_chain_values[chain_index].split('#')
                nodes_enumeration = enumerate(graph_nodes) if is_root else enumerate(graph_nodes[1:], start=1)
                for node_index, service_name in nodes_enumeration:
                    columns = get_node_columns(node_index)
                    generated_graphs[(graph_index, ts)][service_name] = {
                        c.replace(f'_{node_index}', '', 1): value
                        for c, value in row[columns].items()
                    }

        # generate the root chains
        curr_chain_to_generate: List[Tuple[int, int, int, Optional[dict]]] = []
        for (graph_index, ts), chains in root_chains_to_generate.items():
            for chain in chains:
                curr_chain_to_generate.append((ts, graph_index, chain))
        
        print(f"Generating {len(curr_chain_to_generate)} root chains")
        generate_bulk(is_root=True, curr_chain_to_generate=curr_chain_to_generate)

        # generate the chained chains
        print(f"Generating {len(chains_to_generate)} chained chains")
        while any(chains_to_generate.values()):
            curr_chain_to_generate: List[Tuple[int, int, int, Optional[dict]]] = []
            for (graph_index, ts), chains in chains_to_generate.items():
                if len(chains) > 0:
                    any_advancement = False
                    for chain_index in chains.copy():
                        row_nodes = self.all_chain_values[chain_index].split('#')
                        if row_nodes[0] in generated_graphs[graph_index, ts]:
                            curr_chain_to_generate.append((ts, graph_index, chain_index))
                            chains.remove(chain_index)
                            any_advancement = True
                    if not any_advancement:
                        raise Exception(f"Could not generate any chain for graph {graph_index} - no trigger was found")
            generate_bulk(is_root=False, curr_chain_to_generate=curr_chain_to_generate)

        os.makedirs(target_dir_path, exist_ok=True)
        output_file = open(os.path.join(target_dir_path, "generated.json"), "w")
        for (graph_index, ts), data in generated_graphs.items():
            tx = self._component_data_to_tx(graph_index, data, tx_start_time=ts, index_to_node=index_to_node)
            if tx:
                output_file.write(tx + ",\n")

    @staticmethod
    def get(gen_t_config: GenTConfig, is_roll: bool = False) -> "MetadataGenerator":
        return MetadataGenerator(
            gen_t_config,
            is_roll=is_roll
        )
