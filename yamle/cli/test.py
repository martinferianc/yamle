import logging
import os
from argparse import ArgumentParser
from typing import Optional

import pytorch_lightning as pl

import yamle.utils.file_utils as utils
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
from yamle.utils.cli_utils import add_shared_args, add_test_args
from yamle.utils.tracing_utils import trace_input_output_shapes

logging = logging.getLogger("pytorch_lightning")


def evaluate(
    args: ArgumentParser,
    experiment_name: Optional[str] = None,
    overwrite: bool = True,
    overwrite_results: bool = True,
) -> None:
    """This function evaluates a model.

    Args:
        args (ArgumentParser): The arguments of the experiment.
        experiment_name (Optional[str], optional): The name of the experiment. Defaults to None.
        overwrite (bool, optional): Whether to overwrite the model, results, args and log files. Defaults to True.
        overwrite_results (bool, optional): Whether to overwrite the results file. Defaults to True.
    """
    # Set seed
    pl.seed_everything(args.seed, workers=True)

    # Overwrite trainer mode flag
    args.trainer_mode = "eval"

    # Create experiment structure
    if experiment_name is None:
        experiment_name = utils.get_experiment_name(args, mode="eval")

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

    # Prepare the tester and its arguments
    tester_kwargs = prepare_method_kwargs(args, datamodule)
    tester = method_factory(args.method)(
        model=model, loss=loss, regularizer=regularizer, **tester_kwargs
    )
    logging.info("Tester: %s", tester)

    # Create trainer
    trainer_kwargs = prepare_test_trainer_kwargs(args, datamodule)
    trainer = trainer_factory(args.trainer)(**trainer_kwargs, method=tester)
    logging.info("Trainer: %s", trainer)

    pruner_kwargs = prepare_pruner_kwargs(args)
    pruner = pruner_factory(args.pruner)(**pruner_kwargs)
    logging.info("Pruner: %s", pruner)

    quantizer_kwargs = prepare_quantizer_kwargs(args)
    quantizer = quantizer_factory(args.quantizer)(**quantizer_kwargs)
    logging.info("Quantizer: %s", quantizer)

    # The loading of the model is done with respect to the tester
    # which might have done some changes to the model
    tester.on_before_method_load()
    utils.load_method(tester, args.load_path)
    tester.on_after_method_load()
    tester.on_before_model_load()
    utils.load_model(tester.model, utils.model_file(args.load_path))
    logging.info("Model loaded from %s", args.load_path)
    tester.on_after_model_load()

    logging.info("Tracing input and output shapes")
    tester.eval()
    trace_input_output_shapes(tester)

    logging.info("Performing pruning")
    pruner(tester.model)
    pruner.summary(tester.model)

    logging.info("Performing quantization")
    quantizer(trainer, tester)

    # Analysing the method or the model
    logging.info("Analysing model")
    tester.analyse(args.save_path)

    results: Dict[str, Any] = (
        {}
        if not os.path.exists(utils.results_file(args.save_path))
        else utils.load_results(args.save_path)
    )
    logging.info("Inferring model complexity for FLOPS and Params")
    model_complexity(tester.model, tester, trainer.devices, results=results)
    
    logging.info("Testing model")
    trainer.test(results)

    logging.info("Recovering the model after quantization")
    quantizer.recover(tester)

    logging.info("Recovering the model after pruning")
    pruner.recover(tester.model)

    logging.info("Results: %s", results)

    # Save results
    if not args.no_saving:
        utils.save_experiment(
            args.save_path,
            args,
            tester,
            results,
            overwrite=overwrite,
            overwrite_results=overwrite_results,
        )

    logging.info("Experiment completed: %s", experiment_name)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser = add_shared_args(parser)
    parser = add_test_args(parser)

    args = parser.parse_known_args()[0]

    tester = method_factory(args.method)
    model = model_factory(args.model)
    loss = loss_factory(args.loss)
    regularizer = regularizer_factory(args.regularizer)
    datamodule = data_factory(args.datamodule)
    trainer = trainer_factory(args.trainer)
    pruner = pruner_factory(args.pruner)
    quantizer = quantizer_factory(args.quantizer)

    parser = tester.add_specific_args(parser)
    parser = model.add_specific_args(parser)
    parser = loss.add_specific_args(parser)
    parser = regularizer.add_specific_args(parser)
    parser = datamodule.add_specific_args(parser)
    parser = trainer.add_specific_args(parser)
    parser = pruner.add_specific_args(parser)
    parser = quantizer.add_specific_args(parser)

    args = parser.parse_args()
    args = utils.parse_args(args)
    evaluate(args)
