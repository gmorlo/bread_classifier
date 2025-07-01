import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
  model: A target PyTorch model to save.
  target_dir: A directory for saving the model to.
  model_name: A filename for the saved model. Should include
    either ".pth" or ".pt" as the file extension.

  Example usage:
  save_model(model=model_0,
              target_dir="models",
              model_name="05_going_modular_tingvgg_model.pth")
  """
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(), f=model_save_path)

def load_model(model: torch.nn.Module,
                target_dir: str,
                model_name: str) -> torch.nn.Module:
  """Loads a PyTorch model from a target directory.

  Args:
  model: A target PyTorch model to load the state dict into.
  target_dir: A directory for loading the model from.
  model_name: A filename for the saved model. Should include
    either ".pth" or ".pt" as the file extension.

  Returns:
  The loaded PyTorch model with its state dict.

  Example usage:
  loaded_model = load_model(model=model_0,
                            target_dir="models",
                            model_name="05_going_modular_tingvgg_model.pth")
  """
  model_load_path = Path(target_dir) / model_name
  print(f"[INFO] Loading model from: {model_load_path}")
  model.load_state_dict(torch.load(f=model_load_path))
  return model