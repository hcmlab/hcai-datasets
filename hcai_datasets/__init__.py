from pathlib import Path
from hcai_dataset_utils import import_validator

base_dir = Path(__file__).parent

"""Image Datasets"""
if import_validator.validate_installation(
    base_dir / "hcai_affectnet" / "hcai_affectnet.py"
):
    from hcai_datasets import hcai_affectnet

if import_validator.validate_installation(base_dir / "hcai_ckplus" / "hcai_ckplus.py"):
    from hcai_datasets import hcai_ckplus

if import_validator.validate_installation(base_dir / "hcai_faces" / "hcai_faces.py"):
    from hcai_datasets import hcai_faces

"""Audio Datasets"""
if import_validator.validate_installation(
    base_dir / "hcai_audioset" / "hcai_audioset.py"
):
    from hcai_datasets import hcai_audioset

if import_validator.validate_installation(
    base_dir / "hcai_librispeech" / "hcai_librispeech.py"
):
    from hcai_datasets import hcai_librispeech

if import_validator.validate_installation(
    base_dir / "hcai_is2021_ess" / "hcai_is2021_ess.py"
):
    from hcai_datasets import hcai_is2021_ess

"""Nova Datasets"""
from hcai_datasets import hcai_nova_dynamic
