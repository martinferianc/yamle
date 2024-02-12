import argparse
from yamle.data.datamodule import BaseDataModule
from typing import Any, Union, Tuple

from pytorch_lightning import LightningModule
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import requests
import os
import torch
from torch.utils.data import random_split
from yamle.defaults import (
    TEXT_CLASSIFICATION_KEY,
    MEAN_PREDICTION_KEY,
    TRAIN_KEY,
    VALIDATION_KEY,
    TEST_KEY,
    INPUT_KEY,
    TARGET_KEY,
)


class TorchtextClassificationDataModule(BaseDataModule):
    """Data module for the torchvision datasets.

    Args:
        dataset (str): Name of the torchvision dataset. Currently supported are `wiki_text_2`, `wiki_text_103`, `imdb`.
        validation_portion (float): Portion of the training data to use for validation.
        seed (int): Seed for the random number generator.
        data_dir (str): Path to the data directory.
    """

    mean = None
    std = None
    task = TEXT_CLASSIFICATION_KEY

    inputs_dim = None  # This will be the sequence length
    inputs_dtype = torch.long
    outputs_dim = None  # This will be the size of the vocabulary
    outputs_dtype = torch.long

    def __init__(self, dataset: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if dataset not in ["wiki_text_2", "wiki_text_103", "imdb", "shakespeare"]:
            raise ValueError("Dataset not supported.")
        self._dataset = dataset
        self._sequence_length = self.inputs_dim[0]
        self._vocab: torchtext.vocab.Vocab = None

    def _process_data(self, data: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
        """This method is used to tokenize, build the vocabulary and split the sequences into input and target."""
        if self._dataset == "wiki_text_2":
            specials = ["<unk>"]
        elif self._dataset == "wiki_text_103":
            specials = ["<unk>"]
        elif self._dataset == "imdb":
            specials = ["<unk>"]
        elif self._dataset == "shakespeare":
            specials = ["<unk>"]
        tokenizer = get_tokenizer("basic_english")
        if self._vocab is None:
            self._vocab = build_vocab_from_iterator(
                map(tokenizer, data), specials=specials
            )
        self._vocab.set_default_index(self._vocab["<unk>"])
        data = [
            torch.tensor(self._vocab(tokenizer(item)), dtype=torch.long)
            for item in data
        ]
        data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

        inputs = []
        targets = []
        for i in range(0, data.size(0) - 1, self._sequence_length):
            seq_length = min(self._sequence_length, data.size(0) - 1 - i)
            if seq_length != self._sequence_length:
                continue
            inputs.append(data[i : i + self._sequence_length])
            targets.append(data[i + 1 : i + 1 + self._sequence_length])

        inputs = torch.stack(inputs, dim=0)
        targets = torch.stack(targets, dim=0)
        return torch.utils.data.TensorDataset(inputs, targets)

    def prepare_data(self) -> None:
        """Download and prepare the data, the data is stored in `self._train_dataset`, `self._validation_dataset` and `self._test_dataset`."""
        super().prepare_data()
        if self._dataset == "wiki_text_2":
            self._train_dataset = self._process_data(
                torchtext.datasets.WikiText2(root=self._data_dir, split="train")
            )
            self._validation_dataset = self._process_data(
                torchtext.datasets.WikiText2(root=self._data_dir, split="valid")
            )
            self._test_dataset = self._process_data(
                torchtext.datasets.WikiText2(root=self._data_dir, split="test")
            )

        elif self._dataset == "wiki_text_103":
            self._train_dataset = self._process_data(
                torchtext.datasets.WikiText103(root=self._data_dir, split="train")
            )
            self._validation_dataset = self._process_data(
                torchtext.datasets.WikiText103(root=self._data_dir, split="valid")
            )
            self._test_dataset = self._process_data(
                torchtext.datasets.WikiText103(root=self._data_dir, split="test")
            )
        elif self._dataset == "imdb":
            self._train_dataset = self._process_data(
                torchtext.datasets.IMDB(root=self._data_dir, split="train")
            )
            self._test_dataset = self._process_data(
                torchtext.datasets.IMDB(root=self._data_dir, split="test")
            )
        elif self._dataset == "shakespeare":
            data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with open(os.path.join(self._data_dir, "shakespeare.txt"), "w") as f:
                f.write(requests.get(data_url).text)
            with open(os.path.join(self._data_dir, "shakespeare.txt"), "r") as f:
                dataset = self._process_data(f.read().strip().split())
                n = len(dataset)
                train_size = int(n * (1 - self._test_portion))
                test_size = n - train_size
                self._train_dataset, self._test_dataset = random_split(
                    dataset,
                    [train_size, test_size],
                    generator=torch.Generator().manual_seed(self._seed),
                )

        else:
            raise ValueError("Dataset not supported.")

    @torch.no_grad()
    def _get_prediction(
        self,
        tester: LightningModule,
        x: torch.Tensor,
        y: Union[torch.Tensor, int],
        phase: str = TRAIN_KEY,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        super()._get_prediction(tester, x, y, phase)
        x = x.to(tester.device)
        y = y.to(tester.device)
        if phase == TRAIN_KEY:
            output = tester.training_step([x, y], batch_idx=0)
        elif phase == VALIDATION_KEY:
            output = tester.validation_step([x, y], batch_idx=0)
        elif phase == TEST_KEY:
            output = tester.test_step([x, y], batch_idx=0)
        y_hat = output[MEAN_PREDICTION_KEY]
        x = output[INPUT_KEY]
        y = output[TARGET_KEY]
        y_hat = torch.argmax(y_hat, dim=2)
        return y_hat, x, y

    @torch.no_grad()
    def plot(
        self, tester: LightningModule, save_path: str, specific_name: str = ""
    ) -> None:
        """Sample random text sequences from the test set and plot them."""
        # Sample random text sequences from the test set
        train_dataloader = self.train_dataloader()
        inputs, targets = next(iter(train_dataloader))
        outputs = self._get_prediction(tester, inputs, targets, TEST_KEY)[0]

        inputs = inputs.cpu().numpy()
        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()

        for i in range(inputs.shape[0]):
            input = [self._vocab.lookup_token(t) for t in inputs[i]]
            output = [self._vocab.lookup_token(t) for t in outputs[i]]
            target = [self._vocab.lookup_token(t) for t in targets[i]]

            # Write the text sequences to a file
            with open(
                os.path.join(save_path, f"predictions_{specific_name}.txt"), "a"
            ) as f:
                f.write("Input: " + " ".join(input))
                f.write("\n")
                f.write("Output: " + " ".join(output))
                f.write("\n")
                f.write("Target: " + " ".join(target))
                f.write("\n")


class TorchtextClassificationModelWikiText2(TorchtextClassificationDataModule):
    """Data module for the WikiText2 dataset."""

    inputs_dim = (20,)
    outputs_dim = 28782
    targets_dim = 20

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset="wiki_text_2", *args, **kwargs)


class TorchtextClassificationModelWikiText103(TorchtextClassificationDataModule):
    """Data module for the WikiText103 dataset."""

    inputs_dim = (20,)
    outputs_dim = 28782
    targets_dim = 20

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset="wiki_text_103", *args, **kwargs)


class TorchtextClassificationModelIMDB(TorchtextClassificationDataModule):
    """Data module for the IMDB dataset."""

    inputs_dim = (20,)
    outputs_dim = 28782
    targets_dim = 20

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset="imdb", *args, **kwargs)


class Shakespeare(TorchtextClassificationDataModule):
    """Data module for the Shakespeare dataset."""

    inputs_dim = (20,)
    outputs_dim = 28782
    targets_dim = 20

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset="shakespeare", *args, **kwargs)
