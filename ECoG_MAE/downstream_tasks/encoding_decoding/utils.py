from dataclasses import asdict, replace

import numpy as np
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
from scipy import stats

from config import ECoGDataConfig
from downstream_tasks.encoding_decoding.config import (
    EncodingDecodingDataConfig,
    EncodingDecodingExperimentConfig,
)
from downstream_tasks.encoding_decoding.load_signal import EncodingDecodingDataset


def run_encoding_task(
    experiment_config: EncodingDecodingExperimentConfig,
    ecog_data_config: ECoGDataConfig,
    model,
):
    """Run encoding task between word embeddings and neural embeddings.
    
    As of right now these encoding and decoding can use practically the same code. If they start to diverge too much though we
    can split them up more.

    Args:
        experiment_config (EncodingDecodingExperimentConfig): Config for encoding.
        ecog_data_config (ECoGDataConfig): Config for processing data. Should be from model checkpoint.
        model (nn.Module): Callable model on neural data for generating embeddings. Currently just SimpleViT
        inference_device_name (str): Device to run encoding on (i.e. cpu, cuda, etc)

    Returns:
        tuple[np.array, np.array]: (pearson correlations, mean squared prediction error)
    """
    encoding_data_config = merge_data_configs(
        experiment_config.encoding_data_config, ecog_data_config
    )

    dataset = EncodingDecodingDataset(encoding_data_config)

    word_embeddings, neural_embeddings = generate_embedding_dataset(
        dataset,
        model,
        experiment_config.encoding_task_config.embedding_batch_size,
        experiment_config.encoding_task_config.embedding_device,
    )

    predictions = run_regression(
        word_embeddings,
        neural_embeddings,
        experiment_config.encoding_task_config.num_folds,
    )

    correlation_metrics = get_correlation_metrics(neural_embeddings, predictions)
    mspe = np.square(neural_embeddings - predictions).mean()

    return correlation_metrics, mspe


def run_decoding_task(
    experiment_config: EncodingDecodingExperimentConfig,
    ecog_data_config: ECoGDataConfig,
    model,
):
    """Run decoding task between neural embeddings and word embeddings.
    
    As of right now these encoding and decoding can use practically the same code. If they start to diverge too much though we
    can split them up more.

    Args:
        experiment_config (EncodingDecodingExperimentConfig): Config for encoding.
        ecog_data_config (ECoGDataConfig): Config for processing data. Should be from model checkpoint.
        model (nn.Module): Callable model on neural data for generating embeddings. Currently just SimpleViT
        inference_device_name (str): Device to run encoding on (i.e. cpu, cuda, etc)

    Returns:
        tuple[np.array, np.array]: (pearson correlations, mean squared prediction error)
    """
    encoding_data_config = merge_data_configs(
        experiment_config.encoding_data_config, ecog_data_config
    )

    dataset = EncodingDecodingDataset(encoding_data_config)

    word_embeddings, neural_embeddings = generate_embedding_dataset(
        dataset,
        model,
        experiment_config.encoding_task_config.embedding_batch_size,
        experiment_config.encoding_task_config.embedding_device,
    )
    
    # Only change as of now is the order of regression.
    predictions = run_regression(
        neural_embeddings,
        word_embeddings,
        experiment_config.encoding_task_config.num_folds,
    )

    correlation_metrics = get_correlation_metrics(word_embeddings, predictions)
    mspe = np.square(word_embeddings - predictions).mean()

    return correlation_metrics, mspe


def get_correlation_metrics(groundtruth, predicted):
    """
    Calculate summary correlation metrics for two sets of embeddings.

    Args:
    groundtruth (np.array): shape [num_embeddings, embedding_dim]
    predicted (np.array): shape [num_embeddings, embedding_dim]

    Returns:
    dict: A dictionary containing various correlation metrics
    """
    # Flatten the matrices
    gt_flat = groundtruth.flatten()
    pred_flat = predicted.flatten()

    # Calculate overall Pearson correlation
    overall_corr, overall_p = stats.pearsonr(gt_flat, pred_flat)

    # Calculate per-dimension correlations
    dim_corrs = np.array([stats.pearsonr(groundtruth[:, i], predicted[:, i])[0] 
                          for i in range(groundtruth.shape[1])])

    # Calculate per-embedding correlations
    emb_corrs = np.array([stats.pearsonr(groundtruth[i, :], predicted[i, :])[0] 
                          for i in range(groundtruth.shape[0])])

    return {
        "overall_correlation": overall_corr,
        "overall_p_value": overall_p,
        "dimension_correlation": dim_corrs,
        "embedding_correlation": emb_corrs,
        "mean_dimension_correlation": np.mean(dim_corrs),
        "mean_embedding_correlation": np.mean(emb_corrs),
        "median_dimension_correlation": np.median(dim_corrs),
        "median_embedding_correlation": np.median(emb_corrs)
    }


# TODO: Add tests for this.
def run_regression(X: np.array, Y: np.array, num_folds: int) -> np.array:
    """Builds a linear regression model from X->Y using num_folds folds.

    Args:
        X (np.array): Shape [num_examples, num_variables] the independent variable.
        Y (np.array): Shape [num_examples, num_variables] the dependent variable.
        num_folds (int): Number of folds to run training/testing over.

    Returns:
        predicted_values: np.array of shape [num_examples, num_variables] where the ith row corresponds to a prediction of
            the ith in Y, generated when the fold containing the ith row was in the test set.
    """
    # Make folds of data.
    kf = KFold(n_splits=num_folds)

    # Used to track predictions for
    all_predictions = np.zeros_like(Y)

    # TODO: Allow for adding a torch linear head to model which can be used to pass gradients backwards and finetune the
    # model if that's something we choose to do in the future. Using sklearn for now because it's simple and easily
    # runs on cpu. Can also allow for non-linearities in torch model if necessary.
    for train_index, test_index in kf.split(X):
        train_x, test_x = X[train_index], X[test_index]
        train_y = Y[train_index]

        model = make_pipeline(StandardScaler(), LinearRegression())

        model.fit(train_x, train_y)

        predictions = model.predict(test_x)

        all_predictions[test_index, :] = predictions

    return all_predictions


# TODO: Add tests for this.
def generate_embedding_dataset(
    dataset: EncodingDecodingDataset, model: nn.Module, embedding_batch_size: int, device: str
) -> tuple[np.array, np.array]:
    """Gathers word embeddings and generates neural embeddings using model from the dataset.

    Args:
        dataset (EncodingDecodingDataset): dataset used to gather word and neural data.
        model (nn.Module): model used to generate neural embeddings. Expected as of now to output embeddings of shape
            [batch_size, num_tokens, output_dim] where num_tokens is the number of patches for the VideoMAE model, although different
            models could be plugged in here as well. Embeddings are joined together using average pooling to form one summary embedding.
        embedding_batch_size (int): The number of neural examples to pass into the model per-inference. Can speed up inference by
            parallelizing at the cost of RAM or VRAM.
        device (str): The name of the device to run inference on. Model is assumed to already be on this device.

    Returns:
        tuple[np.array, np.array]: (word_embeddings, neural_embeddings) both parallel arrays containing the embeddings for our examples.
    """

    # Setup dataloader and iterate through examples.
    word_embeddings = []
    neural_embeddings = []
    
    def _generate_neural_embeddings(neural_batch: list):
        neural_data = torch.cat(neural_batch)
        neural_data = neural_data.to(device)

        # Model output is shape:
        # [batch_size, num_patches, output_dim]
        model_outputs = model(neural_data)

        # Average pooling, can consider other options in the future.
        pooled_embeddings = torch.mean(model_outputs, dim=1)

        for embedding in pooled_embeddings:
            neural_embeddings.append(embedding.detach().cpu().numpy())

    # Collect data into batches to accelerate inference.
    neural_batch = []
        
    for word_embedding, neural_data in dataset:
        word_embeddings.append(word_embedding)
        batch_ready_neural_data = np.expand_dims(neural_data, 0)
        neural_batch.append(torch.from_numpy(batch_ready_neural_data))

        if len(neural_batch) == embedding_batch_size:
            _generate_neural_embeddings(neural_batch)
            neural_batch = []
            
    # Catch any remaining examples.
    if len(neural_batch) > 0:
        _generate_neural_embeddings(neural_batch)
        neural_batch = []

    word_embeddings = np.array(word_embeddings)
    neural_embeddings = np.array(neural_embeddings)

    return word_embeddings, neural_embeddings


def merge_data_configs(
    encoding_data_config: EncodingDecodingDataConfig, ecog_data_config: ECoGDataConfig
) -> EncodingDecodingDataConfig:
    """Overwrites fields in encoding_data_config with the fields set in ecog_data_config.

    Args:
        encoding_data_config (EncodingDecodingDataConfig): encoding data config which will have fields overwritten.
        ecog_data_config (ECoGDataConfig): ecog data config which contains field to overwrite in encoding_data_config

    Returns:
        EncodingDecodingDataConfig: config with ECoGDataConfig fields overwritten other than original_fs
    """
    # Maintain original_fs from encoding_config because it could be different than the one for pretraining.
    encoding_original_fs = encoding_data_config.original_fs
    updated_config = replace(encoding_data_config, **asdict(ecog_data_config))
    updated_config.original_fs = encoding_original_fs
    return updated_config
