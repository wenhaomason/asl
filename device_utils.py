import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")
    # Test if the device is available
    x = torch.tensor([1.0], device=device)
    print(f"Tensor on {device}: {x}")