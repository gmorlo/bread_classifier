from typing import Dict, List

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training and testing loss curves.
    
    Args: 
        results history dictionary containing lists of values:
        {"train_loss": [...],
        "train_accuracy": [...],
        "test_loss": [...],
        "test_accuracy": [...]}
    """
    pass # TODO: Implement this function to plot loss curves