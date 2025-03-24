import GPUtil

# Check if GPU exists
if len(GPUtil.getAvailable())>0:
    try:
        # Check if pytorch installed
        import torch

        # Check if GPU version of pytorch installed
        if torch.cuda.is_available():
            device = torch.device('cuda')  # Use GPU
            dtype = torch.float32
            
            # Helper function to convert between numpy arrays and tensors
            to_t = lambda array: torch.tensor(array, device=device, dtype=dtype)
            from_t = lambda tensor: tensor.to("cpu").detach().numpy()
    except:
        can_use_GPU = False
else:
    can_use_GPU = False
