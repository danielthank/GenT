import pandas as pd
import torch
import numpy as np

from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from typing import List, Optional, Tuple
from ctgan import CTGAN

class CTGANSynthesizerWrapper:
    """
    A wrapper class for CTGAN that provides a simplified interface for training and 
    sampling synthetic data.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        generator_dim: Tuple[int, ...] = (128, 128),
        discriminator_dim: Tuple[int, ...] = (128, 128),
        generator_lr: float = 2e-4,
        generator_decay: float = 1e-6,
        discriminator_lr: float = 2e-4,
        discriminator_decay: float = 1e-6,
        batch_size: int = 500,
        log_frequency: bool = False,
        verbose: bool = False,
        epochs: int = 300,
        pac: int = 10,
    ):
        """
        Initialize the CTGANSynthesizer.
        
        Args:
            embedding_dim: Size of the random sample passed to the generator.
            generator_dim: Dimensions of the generator layers.
            discriminator_dim: Dimensions of the discriminator layers.
            generator_lr: Learning rate for the generator.
            generator_decay: L2 regularization weight for the generator.
            discriminator_lr: Learning rate for the discriminator.
            discriminator_decay: L2 regularization weight for the discriminator.
            batch_size: Number of data samples to process in each step.
            log_frequency: Whether to use log frequency of categorical levels.
            verbose: Whether to show verbose logs.
            epochs: Number of training epochs.
            pac: Number of samples to group together when applying the discriminator.
        """
        self.model = CTGANSynthesizer(
            embedding_dim=embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            generator_decay=generator_decay,
            discriminator_lr=discriminator_lr,
            discriminator_decay=discriminator_decay,
            batch_size=batch_size,
            log_frequency=log_frequency,
            verbose=verbose,
            epochs=epochs,
            pac=pac,
        )
        self.training_metadata = None
     
    def fit(
        self,
        train_data: pd.DataFrame,
    ):
        """
        Train the CTGAN model.
        
        Args:
            train_data: Training data.
        """
        self.model.fit(train_data)

    def sample(
        self,
        graph_index_list: torch.Tensor,
        chain_index_list: torch.Tensor,
        metadata_list: Optional[pd.DataFrame] = None,
        tx_start_time_list: Optional[pd.DataFrame] = None,
        columns: Optional[List[str]] = None,
        normalize_trigger_data: bool = True
    ) -> pd.DataFrame:
        """
        Sample synthetic data from the trained model.
        
        Args:
            graph_index_list: List of graph indices to sample from.
            chain_index_list: List of chain indices to sample from.
            metadata_list: Metadata for conditioning the samples.
            tx_start_time_list: Transaction start times.
            columns: Specific columns to include in the output.
            normalize_trigger_data: Whether to normalize trigger data.
            
        Returns:
            DataFrame containing generated samples.
        """

        # Initial sampling
        sampled_data = self.model.sample(
            graph_index_list=graph_index_list,
            chain_index_list=chain_index_list,
            metadata_list=metadata_list,
            tx_start_time_list=tx_start_time_list,
            columns=columns,
            normalize_trigger_data=normalize_trigger_data,
        )

        # Identify time-related columns
        time_columns = [col for col in sampled_data.columns if 'gapFromParent' in col or 'duration' in col]
        
        if not time_columns:
            # No time columns found, return as is
            return sampled_data
        
        # Create a mask of valid rows (those without negative time values)
        valid_mask = ~sampled_data.apply(
            lambda row: any(row[col] < 0 for col in time_columns), 
            axis=1
        )

        result_data = sampled_data.copy()
        retry_indices = np.where(~valid_mask)[0]
        retry_count = 0

        while len(retry_indices) > 0 and retry_count < 10:
            print(len(retry_indices), "invalid samples found, retrying...")
            retry_graph_indices = graph_index_list[retry_indices]
            retry_chain_indices = chain_index_list[retry_indices]

            if tx_start_time_list is None:
                retry_tx_start_time = None
            elif isinstance(tx_start_time_list, pd.DataFrame):
                retry_tx_start_time = tx_start_time_list.iloc[retry_indices]
            else:
                retry_tx_start_time = tx_start_time_list[retry_indices]

            if metadata_list is None:
                retry_metadata = None
            elif isinstance(metadata_list, pd.DataFrame):
                retry_metadata = metadata_list.iloc[retry_indices]
            else:
                retry_metadata = metadata_list[retry_indices]
            
            # Generate new samples for the invalid indices
            retry_samples = self.model.sample(
                graph_index_list=retry_graph_indices,
                chain_index_list=retry_chain_indices,
                metadata_list=retry_metadata,
                tx_start_time_list=retry_tx_start_time,
                columns=columns,
                normalize_trigger_data=normalize_trigger_data,
            )
            
            # Check which retry samples are valid
            retry_valid_mask = ~retry_samples.apply(
                lambda row: any(row[col] < 0 for col in time_columns), 
                axis=1
            )
            
            # Update the result dataframe with the valid retry samples in their correct positions
            valid_retry_indices = np.where(retry_valid_mask)[0]
            for valid_idx in valid_retry_indices:
                original_idx = retry_indices[valid_idx]
                result_data.iloc[original_idx] = retry_samples.iloc[valid_idx]
            
            # Update the list of indices to retry in the next iteration
            still_invalid_retry_indices = np.where(~retry_valid_mask)[0]
            retry_indices = retry_indices[still_invalid_retry_indices]
            
            retry_count += 1
        
        return result_data


    def save(self, path: str):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model to.
        """
        self.model.save(path)

    @classmethod
    def load(cls, path: str):
        """
        Load a model from a file.
        
        Args:
            path: Path to load the model from.
            
        Returns:
            Loaded CTGANSynthesizer instance.
        """
        synthesizer = cls()
        synthesizer.model = CTGAN.load(path)
        return synthesizer