'''
    code by Brandon Theodorou
    Original GPT-2 Paper and repository here: https://github.com/openai/gpt-2
    Original GPT-2 Pytorch Model: https://github.com/huggingface/pytorch-pretrained-BERT
    GPT-2 Pytorch Model Derived From: https://github.com/graykode/gpt-2-Pytorch
'''
class GPTConfig(object):
    def __init__(
            self,
            total_vocab_size=6871,
            code_vocab_size=6841,
            label_vocab_size=25,
            special_vocab_size=5, # start, start visits, end visit, end record, pad
            n_positions=750,
            n_ctx=700,
            n_embd=384,
            n_layer=3,
            n_head=4,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            batch_size=48,
            epoch=50,
            lr=1e-4,
    ):
        self.total_vocab_size = total_vocab_size
        self.code_vocab_size = code_vocab_size
        self.label_vocab_size = label_vocab_size
        self.special_vocab_size = special_vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr