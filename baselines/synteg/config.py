class SyntegConfig(object):
    def __init__(self):
        self.embedding_dim = 112
        self.word_embedding_dim = 80
        self.attention_size = 128
        self.ff_dim = 128
        self.max_num_visit = 48
        self.max_length_visit = 80
        self.num_head = 4
        self.code_vocab_dim = 6841
        self.label_vocab_dim = 25
        self.vocab_dim = self.code_vocab_dim + self.label_vocab_dim + 2 # Plus the start and end tokens
        self.head_dim = 32
        self.lstm_dim = 512
        self.n_layer = 3
        self.condition_dim = 256
        self.dependency_batchsize = 40
        self.args = [-1, self.max_num_visit, self.max_length_visit, self.num_head, self.head_dim]
        self.z_dim = 128
        self.g_dims = [256, 256, 512, 512, 512, 512, self.vocab_dim]
        self.d_dims = [256, 256, 256, 128, 128, 128]
        self.gan_batchsize = 2500
        self.gp_weight = 10