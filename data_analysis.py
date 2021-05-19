import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

def get_recall(confusion_matrix):
  return torch.diag(confusion_matrix) / confusion_matrix.sum(dim = 1)

def get_precision(confusion_matrix):
  return torch.diag(confusion_matrix) / confusion_matrix.sum(dim = 0)

def get_accuracy(confusion_matrix):
  return torch.diag(confusion_matrix).sum() / confusion_matrix.sum()

def plot_confusion_matrix(confusion_matrix, all_labels = None):
  if all_labels is None:
    all_labels = ["Country", "Folk", "Jazz", "Hip-Hop", "Metal", "Pop"]

  all_labels.sort()

  df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(),
                     index = all_labels,
                     columns = all_labels)

  plt.figure(figsize = (10,7))
  sn.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
  plt.ylabel("Actual")
  plt.xlabel("Predicted")

def plot_statistics(confusion_matrix, all_labels = None):
  if all_labels is None:
    all_labels = ["Country", "Folk", "Jazz", "Hip-Hop", "Metal", "Pop"]

  all_labels.sort()

  accuracy = get_accuracy(confusion_matrix)
  print("Accuracy:", accuracy.item())

  recall = get_recall(confusion_matrix)
  precision = get_precision(confusion_matrix)

  return pd.DataFrame({"Precision": precision.cpu().numpy(), "Recall": recall.cpu().numpy()}, index = all_labels)
