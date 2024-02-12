from typing import Type
from yamle.data.datamodule import BaseDataModule
from yamle.data.classification import (
    ToyTwoMoonsClassificationDataModule,
    ToyTwoCirclesClassificationDataModule,
    TorchvisionClassificationDataModuleMNIST,
    TorchvisionClassificationDataModuleCIFAR10,
    TorchvisionClassificationDataModuleCIFAR5,
    TorchvisionClassificationDataModuleCIFAR3,
    TorchvisionClassificationDataModuleCIFAR100,
    TorchvisionClassificationDataModuleFashionMNIST,
    TinyImageNetClassificationDataModule,
    TorchvisionClassificationDataModuleSVHN,
    BreastCancerUCIClassificationDataModule,
    AdultIncomeUCIClassificationDataModule,
    CarEvaluationUCIClassificationDataModule,
    CreditUCIClassificationDataModule,
    DermatologyUCIClassificationDataModule,
    PneumoniaMNISTClassificationDataModule,
    DermaMNISTClassificationDataModule,
    BreastMNISTClassificationDataModule,
    BloodMNISTClassificationDataModule,
    ECG5000ClassificationDataModule,
)
from yamle.data.regression import (
    ToyRegressionDataModule,
    ConcreteUCIRegressionDataModule,
    EnergyUCIRegressionDataModule,
    BostonUCIRegressionDataModule,
    TemperatureTimeSeriesDataModule,
    WineQualityUCIRegressionDataModule,
    YachtUCIRegressionDataModule,
    AbaloneUCIRegressionDataModule,
    TelemonitoringUCIRegressionDataModule,
    RetinaMNISTDataModule,
    WikiFaceRegressionDataModule,
    TorchvisionRotationRegressionDataModuleMNIST,
    TorchvisionRotationRegressionDataModuleCIFAR10,
    TorchvisionRotationRegressionDataModuleFashionMNIST,
    TorchvisionRotationRegressionDataModuleSVHN,
    TorchvisionRotationRegressionDataModuleCIFAR100,
    TinyImageNetRotationRegressionDataModule,
)
from yamle.data.segmentation import TorchvisionSegmentationDataModuleCityscapes
from yamle.data.text import (
    TorchtextClassificationModelWikiText2,
    TorchtextClassificationModelWikiText103,
    TorchtextClassificationModelIMDB,
    Shakespeare,
)
from yamle.data.depth import NYUv2DataModule
from yamle.data.reconstruction import ECG5000ReconstructionDataModule

AVAILABLE_DATAMODULES = {
    "mnist": TorchvisionClassificationDataModuleMNIST,
    "cifar3": TorchvisionClassificationDataModuleCIFAR3,
    "cifar5": TorchvisionClassificationDataModuleCIFAR5,
    "cifar10": TorchvisionClassificationDataModuleCIFAR10,
    "cifar100": TorchvisionClassificationDataModuleCIFAR100,
    "svhn": TorchvisionClassificationDataModuleSVHN,
    "fashionmnist": TorchvisionClassificationDataModuleFashionMNIST,
    "tinyimagenet": TinyImageNetClassificationDataModule,
    "wiki_face": WikiFaceRegressionDataModule,
    "pneumoniamnist": PneumoniaMNISTClassificationDataModule,
    "breastmnist": BreastMNISTClassificationDataModule,
    "retinamnist": RetinaMNISTDataModule,
    "dermamnist": DermaMNISTClassificationDataModule,
    "bloodmnist": BloodMNISTClassificationDataModule,
    "toyregression": ToyRegressionDataModule,
    "toymoons": ToyTwoMoonsClassificationDataModule,
    "toycircles": ToyTwoCirclesClassificationDataModule,
    "ecg5000classification": ECG5000ClassificationDataModule,
    "ecg5000reconstruction": ECG5000ReconstructionDataModule,
    "cityscapes": TorchvisionSegmentationDataModuleCityscapes,
    "wikitext2": TorchtextClassificationModelWikiText2,
    "wikitext103": TorchtextClassificationModelWikiText103,
    "imdb": TorchtextClassificationModelIMDB,
    "shakespeare": Shakespeare,
    "concrete": ConcreteUCIRegressionDataModule,
    "energy": EnergyUCIRegressionDataModule,
    "boston": BostonUCIRegressionDataModule,
    "wine": WineQualityUCIRegressionDataModule,
    "yacht": YachtUCIRegressionDataModule,
    "abalone": AbaloneUCIRegressionDataModule,
    "telemonitoring": TelemonitoringUCIRegressionDataModule,
    "breastcancer": BreastCancerUCIClassificationDataModule,
    "adultincome": AdultIncomeUCIClassificationDataModule,
    "carevaluation": CarEvaluationUCIClassificationDataModule,
    "credit": CreditUCIClassificationDataModule,
    "dermatology": DermatologyUCIClassificationDataModule,
    "temperature": TemperatureTimeSeriesDataModule,
    "nyuv2": NYUv2DataModule,
    "rotation_mnist": TorchvisionRotationRegressionDataModuleMNIST,
    "rotation_cifar10": TorchvisionRotationRegressionDataModuleCIFAR10,
    "rotation_fashionmnist": TorchvisionRotationRegressionDataModuleFashionMNIST,
    "rotation_svhn": TorchvisionRotationRegressionDataModuleSVHN,
    "rotation_cifar100": TorchvisionRotationRegressionDataModuleCIFAR100,
    "rotation_tinyimagenet": TinyImageNetRotationRegressionDataModule,
}


def data_factory(data_type: str) -> Type[BaseDataModule]:
    """This function is used to create a data module instance based on the data type."""
    if data_type not in AVAILABLE_DATAMODULES:
        raise ValueError(f"Unknown data type {data_type}.")
    return AVAILABLE_DATAMODULES[data_type]
