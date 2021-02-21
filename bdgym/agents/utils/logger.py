"""A logger helper class """
import time
import pathlib
import os
import os.path as osp
import yaml

from torch.utils.tensorboard import SummaryWriter


AGENT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
DEFAULT_DATA_DIR = osp.join(AGENT_DIR, 'data')

# creates default data directory if it doesn't exist already
if not osp.exists(DEFAULT_DATA_DIR):
    os.mkdir(DEFAULT_DATA_DIR)


class RLLogger(SummaryWriter):
    """RL logger class """

    def __init__(self, env_name, algo_name=None):
        self.env_name = env_name
        self.algo_name = algo_name
        self.save_dir = self.create_log_dir()
        super().__init__(log_dir=self.save_dir)

    def create_log_dir(self):
        """Create directory where logged data is stored """
        tstamp = time.strftime("%Y%m%d-%H%M")
        save_dir = osp.join(
            DEFAULT_DATA_DIR,
            f"{self.algo_name}_{self.env_name}_{tstamp}"
        )
        self.make_dir(save_dir)
        return save_dir

    def get_save_path(self, filename=None, ext=None):
        """Get save path for logger """
        if filename is None:
            tstamp = time.strftime("%Y%m%d-%H%M")
            filename = f"{self.algo_name}_{self.env_name}_{tstamp}"
        return self.generate_file_path(self.save_dir, filename, ext)

    def save_config(self, cfg):
        """Save algorithm config info """
        cfg_file = self.get_save_path("config", "yaml")
        self.write_yaml(cfg_file, cfg)

    @staticmethod
    def make_dir(dir_path):
        """Create a new dir at dir path """
        if osp.exists(dir_path):
            print(f"WARNING: dir {dir_path} already exists.")
        pathlib.Path(dir_path).mkdir(exist_ok=True)

    @staticmethod
    def generate_file_path(parent_dir, file_name, extension):
        """Generates a full file path from a parent directory,
        file name and file extension.
        """
        if extension[0] != ".":
            extension = "." + extension
        return osp.join(parent_dir, file_name + extension)

    @staticmethod
    def write_yaml(file_path, data):
        """Write a dictionary to yaml file """
        with open(file_path, "w") as fout:
            yaml.dump(data, fout)
