"""
Trains a character-level language model.
"""


import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
from mingpt.bpe import BPETokenizer

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

# -----------------------------------------------------------------------------

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

def invert_model(model, tokenizer, target_sentence,config,training_steps,lr):
    model.eval()
    device = next(model.parameters()).device

    tok_target_sentence = tokenizer(target_sentence)
    tok_target_sentence = tok_target_sentence.to(device)
    batch, block_size, n_embd = 1, len(tok_target_sentence[0]), config.model.n_embd

    input_vector = torch.randn((batch, block_size, n_embd), requires_grad=True,device=device)
    optimizer = torch.optim.Adam([input_vector], lr=lr)
    # inversion_process
    for _ in range(training_steps):
        logits  = model.token_embb_bypass_forward(input_vec=input_vector)

        # compute the loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tok_target_sentence.view(-1), ignore_index=-1)

        # backpropagate the gradients
        loss.backward()

        # update the input vector
        optimizer.step()

        # zero the gradients
        optimizer.zero_grad()
    return input_vector

def batch_end_callback(trainer):

    if trainer.iter_num % 10 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")


# -----------------------------------------------------------------------------

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

    # load model
    model.load_state_dict(torch.load(MODEL_PATH))

    # Q2
    target_sentence = "I am a little squirrel holding a walnut"
    output_vector = invert_model(model, tokenizer, target_sentence, config, training_steps=1000, lr=0.1)
    # check if the output vector is the same as the target sentence
    output_logits = model.token_embb_bypass_forward(output_vector)
    output_token_indices = torch.argmax(output_logits,dim=-1)
    output_sentence = tokenizer.decode(output_token_indices[0])
    print(output_sentence)

    # Q3/Q4
    # ar_generation(model, tokenizer, config, target_sentence, 1000, 0.1)
    model.eval()
    sentence = "I am a squirrel holding a ball and I love"

    tok_sentence = tokenizer(sentence)
    tok_sentence = tok_sentence#.to(device)
    output_idx, attention_scores = model.generate(tok_sentence, 1, temperature=1.0, do_sample=True, top_k=10)
    # Average the attention scores over the different attention heads
    attention_scores = attention_scores.mean(dim=1)
    # extract attention scores for the last word (12th)
    att_scores_generated_word = attention_scores[:,10,:]

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 10))
    sns.heatmap(att_scores_generated_word, cmap='viridis',
                xticklabels=sentence.split(" "),
                yticklabels=sentence.split(" "))
    plt.show()



    # Q5
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available
    model.to(device)
    # Sample a sentence from the model

    sentence = "I am a squirrel holding "

    input_ids = tokenizer(sentence).to(device)
    output_idx, _ = model.generate(input_ids, max_new_tokens=10, do_sample=True, top_k=None)
    # Get the log probabilities from the model
    with torch.no_grad():
        logits, _ = model(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)

    # Compute the log-probability score of the sentence
    log_prob_score = 0
    for i in range(1, input_ids.shape[1]):
        log_prob_score += log_probs[0, i - 1, input_ids[0, i]].item()

    print(f'Log-probability score of the sentence: {log_prob_score}')