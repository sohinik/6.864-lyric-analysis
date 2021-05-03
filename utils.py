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


    
# import transformers
# from multiprocessing import Pool
# from tqdm import tqdm, trange

# tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-cased')

# def proc_line_init(tokenizer_for_data):
#     global tokenizer
#     tokenizer = tokenizer_for_data

# def proc_line(sentence):

#     sentence_ids = tokenizer.encode(sentence, verbose=False)
#     return sentence_ids

# def preproc(data, threads, tokenizer):

#     with Pool(threads, initializer=proc_line_init, initargs=(tokenizer,)) as p:
#         data_ids = list(tqdm(p.imap(proc_line, data), total=len(data)))
    
#     return data_ids

# train_ids = preproc(train_dict["lyrics"], 16, tokenizer)
# test_ids =  preproc(test_dict["lyrics"], 16, tokenizer)