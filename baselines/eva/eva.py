import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation, **kwargs)

    def forward(self, input):
        return self.conv(input)[:,:,:-self.conv.padding[0]]

def connector(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.hidden_dim = config.n_embd
        self.embedding_matrix = nn.Linear(config.total_vocab_size, config.n_embd, bias=False)
        self.lstm = nn.LSTM(input_size=config.n_embd,
                            hidden_size=config.n_embd,
                            num_layers=config.n_lstm_layer,
                            bidirectional=True,
                            batch_first=True)
        self.latent_encoder = nn.Linear(2*config.n_embd, 2*config.latent_dim)

    def forward(self, input, lengths):
        visit_emb = self.embedding_matrix(input)
        packed_input = pack_padded_sequence(visit_emb, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out_forward = output[range(len(output)), lengths - 1, :self.hidden_dim]
        out_reverse = output[:, 0, self.hidden_dim:]
        out_combined = torch.cat((out_forward, out_reverse), 1)
        mean_logvar = self.latent_encoder(out_combined)
        return mean_logvar

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose1d(config.latent_dim, 64, 4, stride=2)
        self.deconv2 = nn.ConvTranspose1d(64, 64, 3, stride=2)
        self.deconv3 = nn.ConvTranspose1d(64, 64, 3, stride=2)
        self.deconv4 = nn.ConvTranspose1d(64, 128, 3, stride=3)
        self.causal_conv1 = CausalConv1d(128, 256, 5, dilation=2)
        self.causal_conv2 = CausalConv1d(256, 512, 5, dilation=2)
        self.causal_conv3 = CausalConv1d(512, 4096, 5, dilation=2)
        self.causal_conv4 = CausalConv1d(4096, config.total_vocab_size, 5, dilation=2)

    def forward(self, input):
        input = input.unsqueeze(2)
        out = self.deconv1(input)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.causal_conv1(out)
        out = self.causal_conv2(out)
        out = self.causal_conv3(out)
        out = self.causal_conv4(out)
        out = out.transpose(1, 2)
        return out

class Eva(nn.Module):
    def __init__(self, config):
        super(Eva, self).__init__()
        self.latent_dim = config.latent_dim
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, input_visits, input_lengths, ehr_labels=None, ehr_masks=None, pos_loss_weight=None, kl_weight=1): #kl_weight 0.1 to 1 over a couple epochs
        mean_logvar = self.encoder(input_visits, input_lengths)
        mu = mean_logvar[:,:self.latent_dim]
        log_var = mean_logvar[:,self.latent_dim:]
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        decoder_inputs = connector(mu, log_var)
        code_logits = self.decoder(decoder_inputs)
        sig = nn.Sigmoid()
        code_probs = sig(code_logits)
        if ehr_labels is not None:    
            shift_probs = code_probs[..., :-1, :].contiguous()
            shift_labels = ehr_labels[..., 1:, :].contiguous()
            loss_weights = None
            if pos_loss_weight is not None:
                loss_weights = torch.ones(shift_probs.shape, device=code_probs.device)
                loss_weights = loss_weights + (pos_loss_weight-1) * shift_labels
            if ehr_masks is not None:
                shift_probs = shift_probs * ehr_masks
                shift_labels = shift_labels * ehr_masks
                if pos_loss_weight is not None:
                    loss_weights = loss_weights * ehr_masks

            bce = nn.BCELoss(weight=loss_weights)
            rc_loss = bce(shift_probs, shift_labels)
            loss = rc_loss + kl_weight * kl_loss
            return loss, shift_probs, shift_labels
        
        return code_probs

    def sample(self, batch_size, device):
        decoder_inputs = torch.randn((batch_size, self.latent_dim)).to(device)
        code_logits = self.decoder(decoder_inputs)
        sig = nn.Sigmoid()
        code_probs = sig(code_logits)
        return code_probs

    def marginal_log_likelihood(self, input_ehr, input_lens, input_mask, num_samples):
        bs = input_ehr.size(0)
        mean_logvar = self.encoder(input_ehr, input_lens)
        mu = mean_logvar[:,:self.latent_dim]
        log_var = mean_logvar[:,self.latent_dim:]

        rep_mu = mu.unsqueeze(1).repeat(1,num_samples,1)
        rep_log_var = log_var.unsqueeze(1).repeat(1,num_samples,1)
        rep_sigma = torch.exp(0.5 * rep_log_var)

        latent_samples = connector(rep_mu, rep_log_var)
        latent_samples = latent_samples.reshape((bs * num_samples, self.latent_dim))
        rep_mu = rep_mu.reshape((bs * num_samples, self.latent_dim))
        rep_sigma = rep_sigma.reshape((bs * num_samples, self.latent_dim))

        log2pi = np.log(2*np.pi)
        logp_z = -log2pi * self.latent_dim / 2 - torch.sum(torch.square(latent_samples), axis=1) / 2
        logq_z_x = -log2pi * self.latent_dim / 2 - torch.sum(torch.square((latent_samples - rep_mu) / rep_sigma) + 2 * torch.log(rep_sigma), axis=1) / 2
        code_logits = self.decoder(latent_samples)
        sig = nn.Sigmoid()
        code_probs = sig(code_logits)
        code_probs = code_probs[..., :-1, :].contiguous() 
        ehr_labels = input_ehr[..., 1:, :].contiguous()
        ehr_labels = ehr_labels.repeat((num_samples, 1, 1))
        logp_x_z = torch.sum((ehr_labels * torch.log(code_probs) + (1 - ehr_labels) * torch.log(1 - code_probs)) * input_mask.repeat((num_samples, 1, 1)), axis=(1,2)) # bs*ns
        logp_x = logp_x_z + logp_z - logq_z_x
        logp_x = logp_x.reshape(bs, num_samples)
        m, _ = torch.max(logp_x, dim=1, keepdim=True)
        logprob = m + torch.log(torch.mean(torch.exp(logp_x - m), axis=1, keepdim=True))
        return torch.sum(logprob)