import torch
import torch.nn as nn
import torch.nn.functional as F
from .temporal_shift import TemporalShift # Import TSM

# --- Optional: Attention Layer ---
class AttentionLayer(nn.Module):
    """Simple Self-Attention Layer (Optional)."""
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1) # Apply softmax over the time dimension
        )

    def forward(self, x):
        # x shape: (N, T, H)
        weights = self.attention(x) # (N, T, 1)
        # Weighted sum: (N, T, H) * (N, T, 1) -> sum over T -> (N, H)
        context = torch.sum(x * weights, dim=1)
        return context, weights

# --- Optional: Focal Loss ---
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N, C), targets: (N,)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss) # Probability of the correct class
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Optional: Residual GRU Block ---
class ResidualGRUBlock(nn.Module):
    """GRU Block with Residual Connection."""
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0, bidirectional=False):
        super(ResidualGRUBlock, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0,
                          bidirectional=bidirectional)
        self.bidirectional = bidirectional
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Linear layer for residual connection if dimensions don't match
        if input_dim != gru_output_dim:
            self.residual_proj = nn.Linear(input_dim, gru_output_dim)
        else:
            self.residual_proj = None
        self.layer_norm = nn.LayerNorm(gru_output_dim)

    def forward(self, x, hx=None):
        # x shape: (N, T, input_dim)
        gru_out, h_n = self.gru(x, hx)
        # gru_out shape: (N, T, hidden_dim * num_directions)

        # Residual connection
        if self.residual_proj:
            residual = self.residual_proj(x)
        else:
            residual = x

        # Ensure residual has the same sequence length T
        # If GRU output length differs (unlikely with batch_first=True and no packing)
        # Adjust residual slicing if necessary
        
        out = self.layer_norm(gru_out + residual)
        return out, h_n

# --- Main TSM-GRU Model ---
class TSM_GRU(nn.Module):
    def __init__(self, input_dim=51, hidden_dim=128, num_classes=4,
                 num_layers_gru=1, tsm_segments=8, tsm_shift_div=8,
                 dropout=0.2, bidirectional=False, use_attention=False):
        super(TSM_GRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers_gru = num_layers_gru
        self.tsm_segments = tsm_segments
        self.tsm_shift_div = tsm_shift_div
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Temporal Shift Module
        # Note: TSM is often applied within CNN blocks. Applying before GRU.
        # Ensure input_dim is divisible by tsm_shift_div
        if input_dim % tsm_shift_div != 0:
             print(f"Warning: input_dim ({input_dim}) not divisible by tsm_shift_div ({tsm_shift_div}). TSM might not work as expected.")
        self.tsm = TemporalShift(n_segment=tsm_segments, n_div=tsm_shift_div, inplace=False)

        # GRU Layer(s) - Using Residual Block for potential benefit
        self.gru_block = ResidualGRUBlock(input_dim, hidden_dim, num_layers=num_layers_gru,
                                          dropout=dropout, bidirectional=bidirectional)

        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Optional Attention Layer
        if self.use_attention:
            self.attention_layer = AttentionLayer(gru_output_dim)
            fc_input_dim = gru_output_dim # Attention output is (N, H)
        else:
            self.attention_layer = None
            fc_input_dim = gru_output_dim # Use last hidden state or avg pooling

        # Classifier Head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x):
        # Input x shape: (N, T, C) where C = input_dim (e.g., 51)
        n_batch, seq_len, feat_dim = x.shape

        # Apply TSM
        # Reshape for TSM if needed, or apply directly if TSM handles (N, T, C)
        # Our TSM implementation assumes (N, T, C)
        x_tsm = self.tsm(x)

        # Pass through GRU block
        # gru_out shape: (N, T, hidden_dim * num_directions)
        # h_n shape: (num_layers * num_directions, N, hidden_dim)
        gru_out, h_n = self.gru_block(x_tsm)

        # Select output for classifier
        if self.use_attention:
            # Apply attention over the sequence output of GRU
            context, _ = self.attention_layer(gru_out) # context shape: (N, gru_output_dim)
            out = context
        else:
            # Option 1: Use the last hidden state
            # h_n is (num_layers * num_directions, N, H)
            # Get the last layer's hidden state
            if self.bidirectional:
                 # Concatenate final forward and backward states of the last layer
                 last_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1) # (N, hidden_dim * 2)
            else:
                 last_hidden = h_n[-1,:,:] # (N, hidden_dim)
            out = last_hidden

            # Option 2: Average pooling over time from GRU output
            # out = gru_out.mean(dim=1) # (N, gru_output_dim)

        # Apply dropout and final classification layer
        out = self.dropout(out)
        logits = self.fc(out) # (N, num_classes)

        return logits

# Example Usage (for testing)
if __name__ == '__main__':
    # Test parameters
    batch_size = 4
    seq_length = 150
    input_features = 51
    num_classes = 4

    # Create model instances
    model_no_attn = TSM_GRU(input_dim=input_features, num_classes=num_classes, use_attention=False, bidirectional=True)
    model_attn = TSM_GRU(input_dim=input_features, num_classes=num_classes, use_attention=True, bidirectional=True)

    # Dummy input
    dummy_input = torch.randn(batch_size, seq_length, input_features)

    # Test forward pass (no attention)
    print("Testing model without attention...")
    output_no_attn = model_no_attn(dummy_input)
    print("Output shape (no attention):", output_no_attn.shape) # Expected: (batch_size, num_classes)
    assert output_no_attn.shape == (batch_size, num_classes)

    # Test forward pass (with attention)
    print("\nTesting model with attention...")
    output_attn = model_attn(dummy_input)
    print("Output shape (with attention):", output_attn.shape) # Expected: (batch_size, num_classes)
    assert output_attn.shape == (batch_size, num_classes)

    print("\nModel tests passed.")

    # Test Focal Loss
    print("\nTesting Focal Loss...")
    focal_loss_fn = FocalLoss()
    dummy_logits = torch.randn(batch_size, num_classes)
    dummy_labels = torch.randint(0, num_classes, (batch_size,))
    loss = focal_loss_fn(dummy_logits, dummy_labels)
    print("Focal Loss value:", loss.item())
    assert loss.item() >= 0

    print("Focal Loss test passed.")
