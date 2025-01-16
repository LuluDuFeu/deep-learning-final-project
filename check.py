from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner, calculate_model_size_mb
import torch

if __name__ == '__main__':
  batch_size = 128
  model = CNNPlanner()
  print(model)

  img = torch.randn(batch_size, 3, 96, 128)
  preds = model.forward(img)
  print(preds.shape)
