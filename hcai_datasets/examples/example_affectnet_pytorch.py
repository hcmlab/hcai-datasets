from configparser import ConfigParser

from torch.utils.data import DataLoader

from hcai_dataset_utils.bridge_pytorch import BridgePyTorch
from hcai_datasets.hcai_affectnet.hcai_affectnet_iterable import HcaiAffectnetIterable

if __name__ == "__main__":

    config = ConfigParser()
    config.read("config.ini")

    iterable = HcaiAffectnetIterable(
        dataset_dir=config["directories"]["data_dir"] + "/AffectNet",
        split="test"
    )
    dataloader = DataLoader(BridgePyTorch(iterable))

    for i, sample in enumerate(dataloader):
        if i > 0:
            break
        print(sample)

