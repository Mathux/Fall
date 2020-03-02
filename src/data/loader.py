from .toxicity import FishToxicity, AquaticToxicity
from .concrete import Concrete
from .superconduct import SuperConduct
from .parkinson_updrs import Parkinson_UPDRS_motor, Parkinson_UPDRS_total


ERROR = "This dataset is not available. Please choose one from this list: {}"


def DatasetLoader(dataname):
    if dataname not in DatasetLoader.datasetList:
        dlist = ",".join(DatasetLoader.datasetList.keys())
        raise NotImplementedError(ERROR.format(dlist))

    settings = DatasetLoader.defaultSettings
    dataclass = DatasetLoader.datasetList[dataname]

    if "settings" in dataclass.__dict__:
        settings.update(dataclass["settings"])
        
    settings["name"] = dataname
    datasets = dataclass().datasets
    return datasets, settings


DatasetLoader.datasetList = {method.name if "name" in method.__dict__ else method.__name__.lower() : method
                             for method in [FishToxicity, AquaticToxicity,
                                            Concrete, SuperConduct,
                                            Parkinson_UPDRS_motor,
                                            Parkinson_UPDRS_total]}

DatasetLoader.defaultSettings = {"classification": False}
