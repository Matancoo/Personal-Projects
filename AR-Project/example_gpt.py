
import torch
from torch.utils.data import Dataset
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
from mingpt.bpe import BPETokenizer
# -----------------------------------------------------------------------------
DATASET_PATH = "/Project1/AutoRegressive/alice_in_wonderland.txt"
MODEL_PATH = "/Project1/AutoRegressive/model/GPT2.pt"
# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'#TODO:change here

    # data
    C.data = TextDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    C.trainer.losses = []
    C.trainer.max_iters = 2000

    return C

class TextDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128 # chunk size
        return C

    def __init__(self, config, data):
        self.config = config
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print("Stuff about data")
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        self.data = data.squeeze(0)

    def get_block_size(self):
        return self.config.block_size


    def __len__(self):
        return len(self.data) - self.config.block_size #TODO: check if this is right

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        # sliding this window one token at a time across the dataset
        chunk = self.data[idx:idx + self.config.block_size + 1]

        # return as tensors
        x = chunk[:-1].clone().detach().long() # tokens used to make prediction
        y = chunk[1:].clone().detach().long()  # token to predict next (shifted chunk by 1)
        return x, y

def batch_end_callback(trainer):

    if trainer.iter_num % 10 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

if __name__ == '__main__':
    # get default config and overrides from the command line, if any
    config = get_config()
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    text = open(DATASET_PATH, 'r').read()
    tokenizer = BPETokenizer()
    tokenized_text = tokenizer(text)
    train_dataset = TextDataset(config.data, tokenized_text)

    # construct the model
    config.model.vocab_size = len(tokenizer.encoder.encoder) #TODO: check if this the right size
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)
    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    # trainer.run()

    model.load_state_dict(torch.load(MODEL_PATH))

    # perform sampling: TODO