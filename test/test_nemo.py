"""
PoC using Nvidia Nemo
"""
import nemo
import nemo.collections.asr as nemo_asr
import numpy as np
import pytorch_lightning as pl
import os
import torch

from nemo.collections.asr.parts.utils.speaker_utils import embedding_normalize
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf
from tqdm import tqdm


def main():
    if not torch.cuda.is_available():
        print("[!] Error: CUDA is not available on your machine or pytorch is not installed correctly.")
        exit()

    os.makedirs("./data", exist_ok=True)
    data_dir = "./data"

    # Download the dataset. This will take a few moments...
    print("******")
    if not os.path.exists(data_dir + '/an4_sphere.tar.gz'):
        an4_url = 'https://dldata-public.s3.us-east-2.amazonaws.com/an4_sphere.tar.gz'  # for the original source, please visit http://www.speech.cs.cmu.edu/databases/an4/an4_sphere.tar.gz 
        an4_path = wget.download(an4_url, data_dir)
        print(f"Dataset downloaded at: {an4_path}")
    else:
        print("Tarfile already exists.")
        an4_path = data_dir + '/an4_sphere.tar.gz'

    # Untar and convert .sph to .wav (using sox)
    tar = tarfile.open(an4_path)
    tar.extractall(path=data_dir)

    print("Converting .sph to .wav...")
    sph_list = glob.glob(data_dir + '/an4/**/*.sph', recursive=True)
    for sph_path in sph_list:
        wav_path = sph_path[:-4] + '.wav'
        cmd = ["sox", sph_path, wav_path]
        subprocess.run(cmd)
    print("Finished conversion.\n******")

    train_manifest = os.path.join(data_dir,'an4/wav/an4_clstk/train.json')
    validation_manifest = os.path.join(data_dir,'an4/wav/an4_clstk/dev.json')
    test_manifest = os.path.join(data_dir,'an4/wav/an4_clstk/dev.json')

    MODEL_CONFIG = os.path.join(NEMO_ROOT,'conf/titanet-large.yaml')
    config = OmegaConf.load(MODEL_CONFIG)
    print(OmegaConf.to_yaml(config))

    print(OmegaConf.to_yaml(config.model.train_ds))
    print(OmegaConf.to_yaml(config.model.validation_ds))

    config.model.train_ds.manifest_filepath = train_manifest
    config.model.validation_ds.manifest_filepath = validation_manifest
    config.model.decoder.num_classes = 74

    print("Trainer config - \n")
    print(OmegaConf.to_yaml(config.trainer))

    # Let us modify some trainer configs for this demo
    # Checks if we have GPU available and uses it
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    config.trainer.devices = 1
    config.trainer.accelerator = accelerator

    # Reduces maximum number of epochs to 5 for quick demonstration
    config.trainer.max_epochs = 10

    # Remove distributed training flags
    config.trainer.strategy = None

    # Remove augmentations
    config.model.train_ds.augmentor=None

    trainer = pl.Trainer(**config.trainer)

    log_dir = exp_manager(trainer, config.get("exp_manager", None))
    # The log_dir provides a path to the current logging directory for easy access
    print(log_dir)

    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel(cfg=config.model, trainer=trainer)

    trainer.fit(speaker_model)

    trainer.test(speaker_model, ckpt_path=None)


if __name__ == '__main__':
    main()

