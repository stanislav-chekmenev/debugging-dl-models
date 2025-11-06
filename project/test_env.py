try:
    import torch

    print("Torch installed successfully. Torch version:", torch.__version__)
except ImportError:
    raise ImportError("Please fix your PyTorch installation.")

try:
    import sklearn

    print("Scikit-learn installed successfully. Scikit-learn version:", sklearn.__version__)
except ImportError:
    raise ImportError("Please fix your Scikit-learn installation.")
