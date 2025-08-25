import torch

def debug_network_state(network, input_tensor, frame, action_taken):
    """Debug network state and outputs"""
    network.eval()
    with torch.no_grad():
        q_values = network(input_tensor.unsqueeze(0))

    print(f"[DEBUG Frame {frame}]")
    print(f"Q-values: {q_values.squeeze().cpu().numpy()}")
    print(f"Action taken: {action_taken}")
    print(f"Max Q-value: {q_values.max().item():.6f}")
    print(f"Min Q-value: {q_values.min().item():.6f}")
    print(f"Q-value range: {(q_values.max() - q_values.min()).item():.6f}")

    # Check if all Q-values are the same
    q_std = q_values.std().item()
    if q_std < 1e-6:
        print("⚠️  WARNING: All Q-values are nearly identical!")

    network.train()
    return q_values.squeeze().cpu().numpy()

def debug_gradients(network, frame):
    """Check gradient magnitudes"""
    total_grad_norm = 0.0
    param_count = 0
    zero_grad_params = 0

    print(f"[GRAD DEBUG Frame {frame}]")
    for name, param in network.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            total_grad_norm += grad_norm ** 2
            param_count += 1

            if grad_norm < 1e-8:
                zero_grad_params += 1

            print(f"  {name}: grad_norm = {grad_norm:.8f}")
        else:
            print(f"  {name}: NO GRADIENT")

    total_grad_norm = total_grad_norm ** 0.5
    print(f"Total gradient norm: {total_grad_norm:.8f}")
    print(f"Zero gradient params: {zero_grad_params}/{param_count}")

    if total_grad_norm < 1e-6:
        print("⚠️  WARNING: Gradients are essentially zero!")

    return total_grad_norm

def debug_loss_computation(q_values, target_q_values, frame):
    """Debug loss computation"""
    print(f"[LOSS DEBUG Frame {frame}]")
    print(f"Q-values shape: {q_values.shape}")
    print(f"Target Q-values shape: {target_q_values.shape}")
    print(f"Q-values: {q_values.detach().cpu().numpy()}")
    print(f"Target Q-values: {target_q_values.detach().cpu().numpy()}")

    diff = (q_values - target_q_values)
    print(f"Difference: {diff.detach().cpu().numpy()}")
    print(f"Difference magnitude: {diff.abs().mean().item():.8f}")

    if diff.abs().mean().item() < 1e-8:
        print("⚠️  WARNING: Q-values and targets are nearly identical!")