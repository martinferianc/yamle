import logging
import os
from argparse import ArgumentParser

import pytorch_lightning as pl

import yamle.utils.file_utils as utils
from yamle.cli.test import evaluate
from yamle.data import data_factory
from yamle.evaluation.metrics.hardware import model_complexity
from yamle.methods import method_factory
from yamle.losses import loss_factory
from yamle.models import model_factory
from yamle.pruning import pruner_factory
from yamle.quantization import quantizer_factory
from yamle.regularizers import regularizer_factory
from yamle.trainers import trainer_factory
from yamle.utils.running_utils import *
from yamle.utils.cli_utils import add_shared_args, add_train_args
from yamle.utils.tracing_utils import trace_input_output_shapes

logging = logging.getLogger("pytorch_lightning")


def train(args: ArgumentParser) -> None:
    # Set seed
    pl.seed_everything(args.seed, workers=True)

    # Create experiment structure
    experiment_name = utils.get_experiment_name(args, mode="train")

    # Create experiment directory
    save_path = os.path.join(args.save_path, experiment_name)
    save_path = utils.create_experiment_folder(save_path, "./yamle")
    args.save_path = save_path

    # Set the logger
    utils.config_logger(args.save_path)
    logging.info("Beginning experiment: %s", experiment_name)
    logging.info("Arguments: %s", args)
    logging.info("Command arguments to reproduce: %s", utils.argparse_to_command(args))

    # Prepare the datamodule and its arguments
    datamodule_kwargs = prepare_datamodule_kwargs(args)
    datamodule = data_factory(args.datamodule)(**datamodule_kwargs)
    datamodule.prepare_data()
    datamodule.setup()

    # Prepare the model and its arguments
    model_kwargs = prepare_model_kwargs(args, datamodule)
    model = model_factory(args.model)(**model_kwargs)
    logging.info("Model: %s", model)

    # Prepare the loss and its arguments
    loss_kwargs = prepare_loss_kwargs(args, datamodule)
    loss = loss_factory(args.loss)(**loss_kwargs)
    logging.info("Loss: %s", loss)

    # Prepare the regularizer and its arguments
    regularizer_kwargs = prepare_regularizer_kwargs(args)
    regularizer = regularizer_factory(args.regularizer)(**regularizer_kwargs)
    logging.info("Regularizer: %s", regularizer)

    # Prepare the method and its arguments
    method_kwargs = prepare_method_kwargs(args, datamodule)
    method = method_factory(args.method)(
        model=model, loss=loss, regularizer=regularizer, **method_kwargs
    )
    logging.info("Method: %s", method)
    logging.info("Tracing input and output shapes")
    trace_input_output_shapes(method)

    # Create trainer
    trainer_kwargs = prepare_trainer_kwargs(args, datamodule)
    trainer = trainer_factory(args.trainer)(**trainer_kwargs, method=method)
    logging.info("Trainer: %s", trainer)

    # Train model
    results = {}
    if args.load_path is not None:
        logging.info(
            f"Loading model from {args.load_path}. Note that, only the model is loaded, not the method."
        )
        method.on_before_model_load()
        method.model = utils.load_model(method.model, utils.model_file(args.load_path))
        method.on_after_model_load()

    # Get the model complexity
    model_complexity(method.model, method, trainer.devices, results=results)

    trainer.fit(results)

    # If trainer has been interrupted e.g. by Ctrl+C terminate the experiment
    if trainer.interrupted:
        logging.info("Training interrupted. Terminating experiment.")
        exit(0)

    # Save all the data
    # The model is saved with respect to the method which may have done some
    # processing on it
    if not args.no_saving:
        utils.save_experiment(save_path, args, method, results, overwrite=True)

    # Evaluate model on the default dataset
    if not args.no_evaluation:
        if args.no_saving:
            raise ValueError(
                "Cannot evaluate a model without saving it first. --no_saving must be set to False."
            )
        args.load_path = args.save_path
        evaluate(args, experiment_name=experiment_name)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser = add_shared_args(parser)
    parser = add_train_args(parser)

    args = parser.parse_known_args()[0]

    method = method_factory(args.method)
    model = model_factory(args.model)
    loss = loss_factory(args.loss)
    regularizer = regularizer_factory(args.regularizer)
    datamodule = data_factory(args.datamodule)
    trainer = trainer_factory(args.trainer)
    pruner = pruner_factory(args.pruner)
    quantizer = quantizer_factory(args.quantizer)

    parser = method.add_specific_args(parser)
    parser = model.add_specific_args(parser)
    parser = loss.add_specific_args(parser)
    parser = regularizer.add_specific_args(parser)
    parser = datamodule.add_specific_args(parser)
    parser = trainer.add_specific_args(parser)
    parser = pruner.add_specific_args(parser)
    parser = quantizer.add_specific_args(parser)

    args = parser.parse_args()
    args = utils.parse_args(args)
    train(args)
