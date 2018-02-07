import os.path as path
from .default_generator import generate
from .map import Map

ROOT = path.abspath(path.join(path.dirname(__file__), ".."))

default_config = Map(
    data_dir=ROOT + "/data",
    train_dir=ROOT + "/train",
    data_file="test.npz",
    generate=generate,
    dimension=1,
    data_size=10000
)
