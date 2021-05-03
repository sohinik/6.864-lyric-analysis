import numpy as np
import pandas as pd

def load_raw_data(filename):
    '''
    Input:
    filename: path to the .csv file name stored as a string

    Output:
    data: a pandas DataFrame storing the data
    '''
    
    return pd.read_csv(filename)

def clean_data(data):
    '''
    Input:
    data: a pandas DataFrame storing the data in 'train.csv

    Output:
    cleaned_train_data: a cleaned version of train_data (see README.md
                        for cleaning details) as a pandas DataFrame
    '''
    # Keep only English songs
    cleaned_data = data[data["Language"] == "en"]

    # Keep only the genre and the lyrics
    cleaned_data = cleaned_data[["Genre", "Lyrics"]]

    # Remove datapoints with missing values
    cleaned_data = cleaned_data[cleaned_data["Genre"] != ""]
    cleaned_data = cleaned_data[cleaned_data["Lyrics"] != ""]

    return cleaned_data

def filter_genres(data, genres=None, num_included=None):
    '''
    Input:
    data: Pandas Dataframe representing the data
    genres (optional): list of accepted genres. default is ["Rock", "Metal", "Hip-Hop", 
                        "Country", "Folk"]. can include:
                        - Rock
                        - Pop            
                        - Metal      
                        - Jazz        
                        - Folk      
                        - Indie      
                        - R&B  
                        - Hip-Hop
                        - Electronic
                        - Country
    num_included (optional): number of each genre included. default is the minimum
                             number of datapoints in each genre

    Output:
    cleaned_genre_data: cleaned version of data with max num_included points of the given genres
    '''
    if genres is None:
        genres = ["Rock", "Metal", "Hip-Hop", "Country", "Folk"]

    if num_included is None:
        num_included = data[data["Genre"].isin(genres)]["Genre"].value_counts().min()

    cleaned_genre_data = None

    for genre in genres:
        sampled_genre_data = data[data["Genre"] == genre].sample(n = num_included)
        if cleaned_genre_data is None:
            cleaned_genre_data = sampled_genre_data
        else:
            cleaned_genre_data = cleaned_genre_data.append(sampled_genre_data)
    
    return cleaned_genre_data

def split_data(data, genres = None, training_ratio = 0.8):
    '''
    Randomly splits all of the data into either the training
    or testing data set such that the training and testing set
    still have an equal number of datapoints between genres

    Input:
    data: a pandas DataFrame containing all of the data
    genres (optional): list of accepted genres. default is ["Rock", "Metal", "Hip-Hop", 
                        "Country", "Folk"]. can include:
                        - Rock
                        - Pop            
                        - Metal      
                        - Jazz        
                        - Folk      
                        - Indie      
                        - R&B  
                        - Hip-Hop
                        - Electronic
                        - Country
    training_ratio (optional): the proportion of data used for training, 
                               rest of data is for testing. Default is 0.8

    Outputs:
    train_data: a pandas DataFrame containing the training_data
    test_data: a pandas DataFrame containing the testing_data
    
    '''
    train_data, test_data = None, None 
    for genre in genres:
        genre_data = data[data["Genre"] == genre]
        num_datapoints = len(genre_data.index)
        indices = np.arange(num_datapoints)
        np.random.shuffle(indices)

        num_training = int(training_ratio * num_datapoints)
        training_indices = indices[ :num_training]
        testing_indices = indices[num_training: ]

        genre_train_data = genre_data.iloc[training_indices]
        genre_test_data = genre_data.iloc[testing_indices]

        train_data = pd.concat((train_data, genre_train_data)) if train_data is not None else genre_train_data
        test_data = pd.concat((test_data, genre_test_data)) if test_data is not None else genre_test_data

    return train_data, test_data

def dataframe_to_dict(df):
    '''
    Inputs: 
    df: a pandas DataFrame created in data_processing.py

    Outputs:
    data_dict:
        {
            "labels": [...], # the correct genres in order
            "lyrics": [...], # the correct lyrics in order
        }
    '''

    return {
        "labels": df["Genre"].tolist(),
        "lyrics": df["Lyrics"].tolist(),
    }

def separate_stanzas_from_dict(data_dict, n = 4):
    '''
    Splits lyrics up into datapoints that contain n lines at a time
    
    Inputs:
    data_dict:
        {
            "labels": [...], # the correct genres in order
            "lyrics": [...], # the correct lyrics in order
        }
    n: the number of lines per new datapoint

    Outputs:
    data_dict: the updated dictionary
        {
            "labels": [...], # the correct genres in order
            "lyrics": [...], # the correct lyrics in order
        }
    '''
    new_lyrics = []
    new_labels = []
    for lyric, label in zip(data_dict["lyrics"], data_dict["labels"]):
        lines = lyric.split("\n")
        for i in range(0, len(lines), n):
            new_lyrics.append( "\n ".join(lines[i: i + n]))
            new_labels.append(label)
    
    return {
        "labels": new_labels,
        "lyrics": new_lyrics,
    }

def separate_stanzas_from_dataframe(data, n = 4):
    '''
    Splits lyrics up into datapoints that contain n lines at a time
    
    Inputs:
    data: pandas DataFrame containing the data
    n (optional): the number of lines per new datapoint. default is 4

    Outputs:
    separated_data: pandas DataFrame with separated stanzas
    '''

    new_lyrics = []
    new_labels = []

    for i, row in data.iterrows():
        genre = row["Genre"]
        lyrics = row["Lyrics"]

        lines = lyrics.split("\n")
        for i in range(0, len(lines), n):
            stanza = "\n ".join(lines[i: i + n])
            
            new_lyrics.append(stanza)
            new_labels.append(genre)

    separated_data = pd.DataFrame({"Genre": new_labels, "Lyrics": new_lyrics}).reindex_like(data)

    return separated_data

def get_data(filename = "data.csv", clean_genre=True, 
             genres=None, num_included=None, 
             num_lines_per_stanza = 4, training_ratio = 0.5):
    '''
    Input:
    filename: path to the .csv file name stored as a string
    clean_genre (optional): a boolean that is True if genres should be dropped
                            and the genre distribution should be uniform, and
                            is False otherwise
    genres (optional): list of accepted genres. default is None
    num_included (optional): number of each genre included. default is None
    num_lines_per_stanza: the number of lines per new datapoint. Default is 4.
    training_ratio (optional): the proportion of data used for training, 
                               rest of data is for testing. Default is 0.8

    Outputs:
    train_dict: a dictionary with keys {"lyrics": [...], "labels": [...]}
    test_dict: a dictionary with keys {"lyrics": [...], "labels": [...]}
    '''
    raw_data = load_raw_data(filename)
    cleaned_data = clean_data(raw_data)
    stanza_data = separate_stanzas_from_dataframe(cleaned_data, num_lines_per_stanza)

    if clean_genre: cleaned_data = filter_genres(stanza_data, genres, num_included)
    train_data, test_data = split_data(cleaned_data, genres, training_ratio)

    train_dict = dataframe_to_dict(train_data)
    test_dict = dataframe_to_dict(test_data)

    return train_dict, test_dict

def save_data(data, filename):
    '''
    Saves a pandas DataFrame to a file on disk

    Inputs:
    data: pandas DataFrame
    filename: name of the file to save to
    '''
    data.to_csv(filename)

if __name__ == "__main__":
    # train_data, test_data = get_data()
    pass

    
