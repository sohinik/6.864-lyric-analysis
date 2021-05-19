import torch
import transformers

import torch.nn as nn

from data_processing import get_data
from torch import cuda

class ModelOutputs:
    def __init__(self, logits = None, loss=None):
        self.logits = logits
        self.loss = loss

class GenreClassificationModel(nn.Module):
  def __init__(self, lm, num_labels, dropout=0.2, num_layers = 1, is_bidirectional = False):
    super(GenreClassificationModel, self).__init__()
    # (batch_size, num_tokens)
    # (batch_size, num_tokens, hidden_size)
    # (batch_size, 1 , hidden_size)
    # (batch_size, 1, num_labels)

    self.lm = lm
    self.dropout = nn.Dropout(dropout)
    self.encoder = nn.GRU(
        input_size  = lm.config.hidden_size,
        hidden_size = lm.config.hidden_size,
        num_layers = num_layers,
        batch_first = True,
        bidirectional = is_bidirectional,
        dropout = dropout
        )
    self.classifier = nn.Linear(lm.config.hidden_size, num_labels)
    self.bidirectional = is_bidirectional
    self.num_layers = num_layers


  def forward(self, input_ids, attn_mask, labels = None):
    '''
    Inputs;
    input_ids: (batch_size, num_tokens) tensor of input_ids
    attn_mask: (batch_size, num_tokens) tensor
    labels (optional): (batch_size,) tensor


    Outputs:
    label_logits: (batch_size, num_labels) tensor of logits
    '''

    lm_outputs = self.lm(input_ids, attn_mask)
    hidden_states = lm_outputs.last_hidden_state

    hidden_states = self.dropout(hidden_states)

    _, hidden_states = self.encoder(hidden_states)

    if not self.bidirectional:
      hidden_states = hidden_states[-1]
    else:
      hidden_states = torch.sum(hidden_states[-2:], dim=0)
    logits = self.classifier(hidden_states)

    loss = None

    if labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

    return ModelOutputs(
        logits = logits,
        loss = loss)


def vectorize_batch(batch_inputs, batch_labels, tokenizer, device = "cuda"):

  batch_encode = tokenizer.batch_encode_plus(
      batch_inputs,
      max_length = 512,
      truncation = True,
      padding = "longest",
      return_attention_mask = True,
      return_tensors = 'pt'
  )

  batch_ids = batch_encode["input_ids"].to(device)
  batch_labels = torch.LongTensor(batch_labels).to(device)
  batch_attn_mask = batch_encode["attention_mask"].to(device)

  return batch_ids, batch_labels, batch_attn_mask

def get_label_id_mappings(all_labels = None):
  if all_labels is None:
    all_labels = ["Country", "Folk", "Jazz", "Hip-Hop", "Metal", "Pop"]

  num_labels = len(all_labels)
  all_labels.sort()
  label_to_id_dict = {label: i for i, label in enumerate(all_labels)}
  id_to_label_dict = {i: label for i, label in enumerate(all_labels)}

  return label_to_id_dict, id_to_label_dict, num_labels

def get_ids_for_labels(labels, label_to_id_dict = None):
  if label_to_id_dict is None:
    label_to_id_dict, _, _ = get_label_id_mappings()

  test_label_ids = list(map(lambda x: label_to_id_dict[x], labels))
  return test_label_ids

def get_id(label, label_to_id_dict = None):
  if label_to_id_dict is None:
    label_to_id_dict, _, _ = get_label_id_mappings()
  return label_to_id_dict[label]

def get_label(id, id_to_label_dict = None):
  if id_to_label_dict is None:
    _, id_to_label_dict, _ = get_label_id_mappings()
  return id_to_label_dict[id]

def make_pretrained_model(num_labels = 6, num_layers = 2, is_bidirectional = True):
  """
  Makes an empty model that fits the pretrained classification in the Drive folder
  with the same default parameters
  """
  lm_pretrained = transformers.AutoModel.from_pretrained('distilbert-base-cased')
  model = GenreClassificationModel(lm_pretrained, num_labels=num_labels, num_layers=num_layers, is_bidirectional=is_bidirectional)
  return model

def evaluate(model, inputs, labels, num_labels = 6, batch_size = 16, device = "cuda"):
  """
  Inputs:
  model: an instance of GenreClassificationModel
  inputs: a list of strings where each string is a lyric
  labels: a list of strings where each string is the correct genre
  num_labels: the total number of labels (leave as default for pretrained model)
  device: the runtime device

  Outputs:
  avg_test_loss: the average test loss over all batches
  confusion_matrix: the confusion matrix
  """
  tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-cased')
  labels = get_ids_for_labels(labels)

  model.eval()

  num_data = len(labels)
  batch_size = 16 # batch_size is entire
  num_batches = 0

  total_loss = 0
  confusion_matrix = torch.zeros((num_labels, num_labels)).to(device)

  for i in range(0, num_data, batch_size):
      batch_inputs = inputs[i: i + batch_size]
      batch_labels = labels[i: i + batch_size]

      batch_ids, batch_labels, batch_attn_mask = vectorize_batch(batch_inputs, batch_labels, tokenizer)

      with torch.no_grad():
        outputs = model(
            batch_ids,
            batch_attn_mask,
            batch_labels
        )

      # Back-propagate the loss signal and clip the gradients
      total_loss += outputs.loss.mean()

      # Update confusion matrix
      logits = outputs.logits
      predictions = torch.argmax(logits, dim = 1)
      for _, label, pred in zip(batch_inputs, batch_labels, predictions):
        confusion_matrix[label, pred] += 1

      num_batches += 1

  avg_test_loss = total_loss / num_batches
  return avg_test_loss, confusion_matrix