import torch
import torch.nn as nn

class TemporalShift(nn.Module):
    def __init__(self, n_segment=8, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place TSM')

    def forward(self, x):
        if x.ndim == 3:  # (N, T, C)
            n_batch, t_dim, c_dim = x.size()
            original_t_dim = t_dim  # Store original t_dim

            padded = False
            if t_dim % self.n_segment != 0:
                padding_size = self.n_segment - (t_dim % self.n_segment)
                padding_tensor = torch.zeros(n_batch, padding_size, c_dim, device=x.device, dtype=x.dtype)
                x = torch.cat((x, padding_tensor), dim=1)
                t_dim = x.shape[1]  # Update t_dim to the new padded dimension
                padded = True

            # Now t_dim is divisible by self.n_segment
            x = x.view(n_batch * self.n_segment, t_dim // self.n_segment, c_dim)
        else:
            raise NotImplementedError("TSM for non-3D tensor not adapted for keypoints here.")

        fold = c_dim // self.fold_div
        if self.inplace:
            x[:, :-1, :fold] = x[:, 1:, :fold]
            x[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
            out = x
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]
            out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        # Reshape back using the potentially padded t_dim
        out = out.view(n_batch, t_dim, c_dim)

        # Truncate back to original_t_dim if padding was applied
        if padded:
            out = out[:, :original_t_dim, :]

        return out

if __name__ == '__main__':
    # Example Usage:
    # Assuming we have sequences of keypoints (Batch, Time, Features)
    # Let Features = 63 (21 keypoints * 3 coords)
    # Let Time = 150 frames
    # Let Batch = 4

    # Test with (N, T, C)
    # N=4, T=160 (divisible by n_segment=8 -> 20 frames per segment), C=64 (divisible by n_div=8)
    # Main TSM class test (how it might be used in a model with segmentation):
    print("Testing TemporalShift class (as used in a model with segmentation):")
    tsm_module = TemporalShift(n_segment=8, n_div=8)
    test_input_main = torch.rand(4 * 8, 160 // 8, 64)  # Simulating (N*num_segments, T_per_segment, C)
    print(f"Input shape for main TSM: {test_input_main.shape}")
    test_output_main = tsm_module(test_input_main)
    # Reshape back to original conceptual shape (N, T, C)
    test_output_main_reshaped = test_output_main.view(4, 160, 64)
    print(f"Output shape for main TSM (reshaped): {test_output_main_reshaped.shape}")

    # A more direct TSM for (N, T, C) where C is feature dim and T is time
    # This version shifts feature channels over time segments.
    class SimpleTSMForSequences(nn.Module):
        def __init__(self, n_segment=8, n_div=8, inplace=False):
            super().__init__()
            self.n_segment = n_segment
            self.n_div = n_div
            self.inplace = inplace

        def forward(self, x):
            # x shape: (N, T, C)
            N, T, C = x.shape
            
            if T % self.n_segment != 0:
                raise ValueError(f"Time dimension ({T}) must be divisible by n_segment ({self.n_segment})")
            if C % self.n_div != 0:
                raise ValueError(f"Channel dimension ({C}) must be divisible by n_div ({self.n_div})")

            # Number of frames per segment
            frames_per_segment = T // self.n_segment
            
            # Reshape to (N, num_segments, frames_per_segment, C)
            x_reshaped = x.view(N, self.n_segment, frames_per_segment, C)
            
            # Number of channels to shift (C // n_div)
            fold = C // self.n_div
            
            if self.inplace:
                shifted_x = x_reshaped  # Operate in-place
            else:
                shifted_x = torch.zeros_like(x_reshaped)
            
            # Shift operation (example: shift 1/n_div of channels)
            # Shift left for the first fold of channels
            shifted_x[:, :-1, :, :fold] = x_reshaped[:, 1:, :, :fold]  # Segments shifted left
            if not self.inplace:  # If not in-place, last segment needs to be copied for this part
                shifted_x[:, -1, :, :fold] = x_reshaped[:, -1, :, :fold]
            
            # Shift right for the second fold of channels
            shifted_x[:, 1:, :, fold:2*fold] = x_reshaped[:, :-1, :, fold:2*fold]  # Segments shifted right
            if not self.inplace:  # If not in-place, first segment needs to be copied for this part
                shifted_x[:, 0, :, fold:2*fold] = x_reshaped[:, 0, :, fold:2*fold]

            # Copy the rest if not in-place
            if not self.inplace:
                shifted_x[:, :, :, 2*fold:] = x_reshaped[:, :, :, 2*fold:]
            elif C > 2 * fold:  # If in-place, only copy if there are more than 2 folds
                pass  # The rest of x_reshaped (which is shifted_x) is already correct
            
            return shifted_x.view(N, T, C)

    print("\nTesting SimpleTSMForSequences:")
    dummy_input_ntc_simple = torch.rand(4, 160, 64)  # N, T, C
    simple_tsm_instance = SimpleTSMForSequences(n_segment=8, n_div=8)
    output_simple_instance = simple_tsm_instance(dummy_input_ntc_simple)
    print(f"Input shape for SimpleTSM: {dummy_input_ntc_simple.shape}, Output shape: {output_simple_instance.shape}")
    
    # The TSM_GRU model will likely use a TSM module like the first one (TemporalShift class),
    # but applied after a linear layer that projects input_dim to something divisible by n_div,
    # and it will handle the sequence segmentation.
    # The provided `train_tsm_gru.py` uses `tsm_segments` and `tsm_shift_div`
    # which implies the TSM is applied in segments, consistent with the main TemporalShift class.
