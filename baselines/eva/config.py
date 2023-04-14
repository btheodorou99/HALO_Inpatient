class EVAConfig(object):
    def __init__(
            self,
            total_vocab_size=6869,
            code_vocab_size=6841,
            label_vocab_size=25,
            special_vocab_size=3,
            n_ctx=57,
            n_embd=768,
            latent_dim=32,
            n_lstm_layer=1,
            n_conv1d_layer=3,
            n_deconv_layer=4,
            dilation_factor=2,
            deconv_factor=3,
            batch_size=128,
            prob_batch_size=4,
            epoch=50,
            lr=1e-4,
            pos_loss_weight=None
    ):
        self.total_vocab_size = total_vocab_size
        self.code_vocab_size = code_vocab_size
        self.label_vocab_size = label_vocab_size
        self.special_vocab_size = special_vocab_size
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.latent_dim = latent_dim
        self.n_lstm_layer = n_lstm_layer
        self.n_conv1d_layer = n_conv1d_layer
        self.n_deconv_layer = n_deconv_layer
        self.dilation_factor = dilation_factor
        self.deconv_factor = deconv_factor
        self.batch_size = batch_size
        self.prob_batch_size = prob_batch_size
        self.epoch = epoch
        self.lr = lr
        self.pos_loss_weight = pos_loss_weight