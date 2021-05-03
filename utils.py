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

# def 
#     '''
#     Splits lyrics up into datapoints that contain n lines at a time
    
#     Inputs:
#     data_dict:
#         {
#             "labels": [...], # the correct genres in order
#             "lyrics": [...], # the correct lyrics in order
#         }
#     n: the number of lines per new datapoint

#     Outputs:
#     data_dict: the updated dictionary
#         {
#             "labels": [...], # the correct genres in order
#             "lyrics": [...], # the correct lyrics in order
#         }
#     '''
#     new_lyrics = []
#     new_labels = []
#     for lyric, label in zip(data_dict["lyrics"], data_dict["labels"]):
#         lines = lyric.split("\n")
#         for i in range(0, len(lines), n):
#             new_lyrics.append( "\n ".join(lines[i: i + n]))
#             new_labels.append(label)
    
#     return {
#         "labels": new_labels,
#         "lyrics": new_lyrics,
#     }
