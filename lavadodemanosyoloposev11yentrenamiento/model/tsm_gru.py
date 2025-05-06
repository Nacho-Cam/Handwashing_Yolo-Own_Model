import torch
import torch.nn as nn
import torch.nn.functional as F
from .temporal_shift import TemporalShift # Import from the same directory

class TSM_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers_gru,
                 tsm_segments=8, tsm_shift_div=8, dropout=0.5,
                 bidirectional=False, use_attention=False, use_batch_norm=False):
        super(TSM_GRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers_gru = num_layers_gru
        self.tsm_segments = tsm_segments
        self.tsm_shift_div = tsm_shift_div # Make sure input_dim is divisible by this
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_batch_norm = use_batch_norm

        # --- Feature Processing Layers ---
        if self.use_batch_norm:
            # BatchNorm1d expects (N, C) or (N, C, L).
            # If input is (N, L, C), we need to permute.
            self.bn_input = nn.BatchNorm1d(input_dim)

        # Temporal Shift Module
        # The TSM module itself expects input that has been segmented.
        # Or, the TSM module can handle the segmentation view internally.
        # The TemporalShift class in temporal_shift.py is designed to take (N, T, C)
        # and handle the view (N*num_segments, T_per_segment, C) internally.
        self.tsm_module = TemporalShift(n_segment=self.tsm_segments, n_div=self.tsm_shift_div, inplace=False)

        # GRU input dimension should be the original feature dimension
        gru_input_dim = input_dim # Corrected: GRU input dim is not divided by tsm_segments

        # GRU Layer
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers_gru,
            batch_first=True,
            dropout=self.dropout_rate if num_layers_gru > 1 else 0,
            bidirectional=bidirectional
        )

        # --- Attention Mechanism (Optional) ---
        if self.use_attention:
            # Simplified attention: a linear layer followed by softmax
            attention_input_features = hidden_dim * 2 if bidirectional else hidden_dim
            self.attention_weights = nn.Linear(attention_input_features, 1)
            # Softmax will be applied over the sequence dimension

        # --- Classifier ---
        classifier_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout_fc = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(classifier_input_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        # 1. Optional Input Batch Normalization
        if self.use_batch_norm:
            # Permute to (batch_size, input_dim, seq_len) for BatchNorm1d
            x = x.permute(0, 2, 1)
            x = self.bn_input(x)
            # Permute back to (batch_size, seq_len, input_dim)
            x = x.permute(0, 2, 1)

        # 2. Temporal Shift Module
        # Ensure seq_len is divisible by tsm_segments for the TSM module's internal reshaping.
        # The HandWashDataset in train_tsm_gru.py pads sequences.
        # If seq_len is still not divisible, pad it here.
        if seq_len % self.tsm_segments != 0:
            padding_len = self.tsm_segments - (seq_len % self.tsm_segments)
            padding = torch.zeros(batch_size, padding_len, self.input_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
            # seq_len = x.shape[1] # Update seq_len if needed later

        x = self.tsm_module(x) # Apply TSM

        # 3. GRU Layer
        # gru_out: (batch_size, seq_len, hidden_dim * num_directions)
        # h_n: (num_layers * num_directions, batch_size, hidden_dim)
        gru_out, h_n = self.gru(x)

        # 4. Optional Attention Mechanism or Last Hidden State
        if self.use_attention:
            # gru_out: (batch, seq, features)
            # Calculate attention scores
            attn_energies = self.attention_weights(gru_out).squeeze(-1) # (batch, seq)
            attn_probs = F.softmax(attn_energies, dim=1) # (batch, seq)
            
            # Weighted sum of GRU outputs
            # attn_probs.unsqueeze(2): (batch, seq, 1)
            context_vector = torch.sum(gru_out * attn_probs.unsqueeze(2), dim=1) # (batch, features)
            output_feature = context_vector
        else:
            # Use the output of the last GRU time step
            # gru_out is (batch, seq, num_directions * hidden_size)
            # We want the features from the last time step.
            output_feature = gru_out[:, -1, :]


        # 5. Classifier
        output_feature = self.dropout_fc(output_feature)
        out = self.fc(output_feature) # (batch_size, num_classes)

        return out

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # Can be a scalar or a tensor of weights for each class
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N, C) logits from the model
        # targets: (N) ground truth labels
        
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss) # p_t
        
        # Calculate Focal Loss
        # If alpha is a scalar, it weights all positive classes.
        # If alpha is a list/tensor, it can weight each class.
        # For multi-class, a common way to use scalar alpha is to apply it as a general weight.
        # Or, if alpha is a tensor of shape (C,), use gather to get alpha_t for each sample.
        
        if isinstance(self.alpha, float): # Scalar alpha
            alpha_t = self.alpha
            focal_term = (1 - pt)**self.gamma
            focal_loss = alpha_t * focal_term * CE_loss
        elif isinstance(self.alpha, torch.Tensor): # Per-class alpha
            # Ensure alpha is on the same device as inputs
            alpha_t = self.alpha.to(inputs.device)[targets] # Get alpha for each target class
            focal_term = (1 - pt)**self.gamma
            focal_loss = alpha_t * focal_term * CE_loss
        else: # Default if alpha is not float or tensor (e.g. None)
            focal_term = (1 - pt)**self.gamma
            focal_loss = focal_term * CE_loss


        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else: # 'none'
            return focal_loss

if __name__ == '__main__':
    # Example Usage
    batch_size = 4
    # seq_len must be divisible by tsm_segments for the TSM module if no internal padding
    # With internal padding in TSM_GRU, any seq_len can be used.
    seq_len = 150 
    input_dim = 63
    hidden_dim = 256
    num_classes = 4
    num_layers_gru = 2
    
    # Ensure tsm_segments and tsm_shift_div are compatible with dimensions
    # input_dim (63) should be divisible by tsm_shift_div (e.g., 3)
    # seq_len (150) should be divisible by tsm_segments (e.g., 10, 6, 5, etc.)
    # The train_tsm_gru.py uses tsm_segments=8, tsm_shift_div=3.
    # 150 is not divisible by 8. The TSM_GRU model now pads internally.
    # 63 is divisible by 3.
    
    tsm_segments_test = 8 
    tsm_shift_div_test = 3

    print(f"Testing with: seq_len={seq_len}, input_dim={input_dim}, tsm_segments={tsm_segments_test}, tsm_shift_div={tsm_shift_div_test}")

    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    dummy_labels = torch.randint(0, num_classes, (batch_size,))

    # --- Test TSM_GRU model ---
    print("\nTesting TSM_GRU model...")
    model = TSM_GRU(input_dim, hidden_dim, num_classes, num_layers_gru,
                    tsm_segments=tsm_segments_test, tsm_shift_div=tsm_shift_div_test,
                    dropout=0.5, bidirectional=True, use_attention=True, use_batch_norm=True)
    
    print(model)
    
    try:
        output = model(dummy_input)
        print(f"TSM_GRU Output shape: {output.shape} (Expected: ({batch_size}, {num_classes}))")
        assert output.shape == (batch_size, num_classes)
        print("TSM_GRU forward pass successful.")
    except Exception as e:
        print(f"Error during TSM_GRU forward pass: {e}")
        import traceback
        traceback.print_exc()

    # --- Test FocalLoss ---
    print("\nTesting FocalLoss (scalar alpha)...")
    criterion_focal_scalar = FocalLoss(alpha=0.25, gamma=2.0)
    logits = output # Use output from model test
    
    try:
        loss_scalar = criterion_focal_scalar(logits, dummy_labels)
        print(f"Focal Loss (scalar alpha): {loss_scalar.item()}")
        print("FocalLoss (scalar alpha) calculation successful.")
    except Exception as e:
        print(f"Error during FocalLoss (scalar alpha) calculation: {e}")

    print("\nTesting FocalLoss (per-class alpha)...")
    alpha_per_class = torch.tensor([0.1, 0.2, 0.3, 0.4]) # Example: one weight per class
    criterion_focal_per_class = FocalLoss(alpha=alpha_per_class, gamma=2.0)
    try:
        loss_per_class = criterion_focal_per_class(logits, dummy_labels)
        print(f"Focal Loss (per-class alpha): {loss_per_class.item()}")
        print("FocalLoss (per-class alpha) calculation successful.")
    except Exception as e:
        print(f"Error during FocalLoss (per-class alpha) calculation: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting FocalLoss (alpha=None)...")
    criterion_focal_no_alpha = FocalLoss(alpha=None, gamma=2.0)
    try:
        loss_no_alpha = criterion_focal_no_alpha(logits, dummy_labels)
        print(f"Focal Loss (no alpha): {loss_no_alpha.item()}")
        print("FocalLoss (no alpha) calculation successful.")
    except Exception as e:
        print(f"Error during FocalLoss (no alpha) calculation: {e}")
