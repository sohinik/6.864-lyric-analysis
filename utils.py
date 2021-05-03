import torch

def save_model(model, filepath):
    '''
    Inputs:
    model: a torch Object to save to a file
    filepath: the path to save to
              e.g.
              "/content/gdrive/MyDrive/6.864/hw4/model.pt"
    '''
    torch.save(model.state_dict(), filepath)
    
def load_model(model, filepath):
    '''
    Inputs:
    model: a torch Object to load and mutate
    filepath: the path to load the model from
              e.g.
              "/content/gdrive/MyDrive/6.864/hw4/model.pt"

    Returns:
    model: torch Object with loaded model in
    '''
    model.load_state_dict(torch.load(filepath))
    return model