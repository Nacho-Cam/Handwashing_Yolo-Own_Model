import torch
import torch.nn as nn

class TemporalShift(nn.Module):
    """
    Temporal Shift Module (TSM).
    Shifts part of the channels along the time dimension.
    """
    def __init__(self, n_segment=8, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')

    def forward(self, x):
        # x shape: (N, T, C) - Assuming Batch first format common with RNNs
        # Or potentially (N, C, T) - Common in CNNs. Let's assume (N, T, C) based on GRU usage.
        # If input is (N, C, T), uncomment the permute lines.

        # x = x.permute(0, 2, 1) # (N, T, C) -> (N, C, T) if needed

        # Or if input is (N, T, C):
        n_batch, t, c = x.size() # <-- CORRECT UNPACKING FOR (N, T, C)
        fold = c // self.fold_div
        
        # Reshape considering segments if needed, but simpler shift works too
        # Assuming t is a multiple of n_segment for simplicity here
        # x = x.view(n_batch, t // self.n_segment, self.n_segment, c)

        out = x.clone() # Use clone if not inplace

        # Shift left (past to future)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        # Shift right (future to past)
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        # Channels fold*2 onwards are not shifted

        # If inplace:
        # if self.inplace:
        #     # Shift left
        #     out[:, :-1, :fold] = x[:, 1:, :fold]
        #     # Shift right
        #     out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        #     return out
        # else:
        #     # Create output tensor
        #     res = torch.zeros_like(x)
        #     # Copy non-shifted part
        #     res[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        #     # Apply shifts
        #     res[:, :-1, :fold] = x[:, 1:, :fold]       # Shift left
        #     res[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold] # Shift right
        #     return res

        # Reshape back if needed
        # return out.view(n_batch, t, c)

        # out = out.permute(0, 2, 1) # (N, C, T) -> (N, T, C) if needed

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(n_segment={self.n_segment}, n_div={self.fold_div}, inplace={self.inplace})'

# Example Usage (for testing)
if __name__ == '__main__':
    # Example Input: Batch=2, Time=16, Channels=64
    # Assume n_segment=8, n_div=8
    tsm = TemporalShift(n_segment=8, n_div=8)
    # Input shape (N, T, C)
    input_tensor = torch.randn(2, 16, 64)
    output_tensor = tsm(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)

    # Check if shapes match
    assert input_tensor.shape == output_tensor.shape

    # Check shifts (example: first segment, first fold)
    # Output's first time step should be zero for the first fold (shifted from t=-1)
    # Output's second time step should be Input's first time step for the first fold
    print("Input [0, 0, :8]:", input_tensor[0, 0, :8])
    print("Output[0, 1, :8]:", output_tensor[0, 1, :8])
    assert torch.allclose(input_tensor[0, 0, :8], output_tensor[0, 1, :8])

    # Check shift right (example: first segment, second fold)
    # Output's first time step should be Input's second time step for the second fold
    print("Input [0, 1, 8:16]:", input_tensor[0, 1, 8:16])
    print("Output[0, 0, 8:16]:", output_tensor[0, 0, 8:16])
    # Note: This assertion might fail due to boundary conditions (t=0 has no t=-1)
    # Let's check t=1 shifted from t=0
    print("Input [0, 0, 8:16]:", input_tensor[0, 0, 8:16])
    print("Output[0, 1, 8:16]:", output_tensor[0, 1, 8:16])
    assert torch.allclose(input_tensor[0, 0, 8:16], output_tensor[0, 1, 8:16])

    print("TSM module test passed.")
