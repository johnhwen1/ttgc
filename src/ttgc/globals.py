import GPUtil

# Check if GPU exists
if len(GPUtil.getAvailable())>0:
    try:
        # Check if pytorch installed
        import torch

        # Check if GPU version of pytorch installed
        if torch.cuda.is_available():
            can_use_GPU = True
    except:
        can_use_GPU = False
else:
    can_use_GPU = False
