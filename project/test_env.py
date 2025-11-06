try:
    import torch
    import sklearn

    print("Torch installed successfully. Torch version:", torch.__version__)
    print("Scikit-learn installed successfully. Scikit-learn version:", sklearn.__version__)
except ImportError:
    raise ImportError("Please fix your PyTorch and/or Scikit-learn installation.")
