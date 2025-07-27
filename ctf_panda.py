from unicodedata import normalize
import numpy as np
from typing import Dict, Optional
import torch

from panda.patchtst.pipeline import PatchTSTPipeline
from panda.utils import (
    apply_custom_style,
    plot_trajs_multivariate, safe_standardize
)

class PandaModel:
    def __init__(self, config: Dict, train_data: Optional[np.ndarray] = None, init_data: Optional[np.ndarray] = None, training_timesteps: Optional[np.ndarray] = None, prediction_timesteps: Optional[np.ndarray] = None, pair_id: Optional[int] = None):
        """
        Initialize the Panda model with the given configuration and training data.
        :param config: Configuration dictionary containing model parameters.
        :param
        train_data: Training data for the model.
        :param
        init_data: warm-start data for parametric regimes.
        :param prediction_time_steps: time instances at which to predict.
        :param training_time_steps: time instances at which to train.
        :param pair_id: Identifier for the data pair.
        """
        # Load configuration parameters
        self.config = config
        self.pair_id = pair_id
        self.dataset_name = config['dataset']['name']
        self.pair_id = pair_id

        # Load training data (need to reshape it for PyKoopman)
        self.train_data = train_data
        # self.train_data = self.train_data.squeeze()
        self.init_data = init_data
        # if self.init_data is not None:
        #     self.init_data = self.init_data.squeeze()
        
        # Load auxiliary dataparameters
        self.prediction_timesteps = prediction_timesteps
        self.training_timesteps = training_timesteps[0]
        self.dt = self.prediction_timesteps[1] - self.prediction_timesteps[0]
        
        


    def train(self):
        """
        Load the Panda model from https://huggingface.co/GilpinLab/panda_mlm/tree/main
        """
        print("PandaModel: loaded as zero-shot model, no training is performed.")

        ## Load MLM trained model from Hugging Face
        self.model_pipeline = PatchTSTPipeline.from_pretrained(
            mode="predict",
            pretrain_path=self.config['model']['weights'],
            device_map="cpu",#"cuda:0",
            torch_dtype=torch.float32,
        )


    def predict(self):
        """        Predict the future states of the system using the trained model.
        :return: Predicted data.
        """

        # For panda, first dimension is time, second dimension is features
        # Initialise the parametric regime
        if self.pair_id in [8,9]:
            context_load = self.init_data
        else:
            context_load = self.train_data

        context = context_load.copy()

        if self.config['model']['normalize']:
            context = safe_standardize(context,context=context_load, axis=0)


        # For PDE datasets the full init/training data is too large as context for the foundation model so need to manually limit it
        context_length = self.config['model'].get('context_length', 128)
        if context_length == -1:
            context_length = float('inf')

        context_length = max(128, context_length)
        if context.shape[0] > context_length:
            context = context[-context_length:, :]


        prediction = self.model_pipeline.predict(
            context=torch.tensor(context).float(),
            prediction_length=self.prediction_timesteps.shape[0],
            limit_prediction_length=False,
            sliding_context=True,
            ).squeeze().cpu().numpy()

        prediction = prediction[:self.prediction_timesteps.shape[0], :] # Panda returns more timesteps than requested, so we slice it

        # Manually fix overflow issues
        prediction[np.isnan(prediction)]=0.0


        if self.config['model']['normalize']:
            return safe_standardize(
                prediction,
                axis=0,
                context=context_load,
                denormalize=True,
            )
        else:
            return prediction