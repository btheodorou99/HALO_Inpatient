import torch
import torch.nn as nn

class LSTMBaseline(nn.Module):
    def __init__(self, config):
        super(LSTMBaseline, self).__init__()
        self.embedding_matrix = nn.Linear(config.total_vocab_size, config.n_embd, bias=False)
        self.lstm = nn.LSTM(input_size=config.n_embd,
                            hidden_size=config.n_embd,
                            num_layers=6,
                            batch_first=True)
        self.ehr_head = nn.Linear(config.n_embd, config.total_vocab_size)

    def forward(self, input_visits, ehr_labels=None, ehr_masks=None, pos_loss_weight=None):
        embeddings = self.embedding_matrix(input_visits)
        hidden_states, _ = self.lstm(embeddings)
        code_logits = self.ehr_head(hidden_states)
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
            loss = bce(shift_probs, shift_labels)
            return loss, shift_probs, shift_labels
        
        return code_probs