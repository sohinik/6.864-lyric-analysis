import numpy as np
import pandas as pd
import random

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

    # Remove datapoints with bad lyrics
    cleaned_data = cleaned_data[not cleaned_data.contains("---") or not cleaned_data.contains("___") or not cleaned_data.contains("|")]

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
    if genres is None:
        genres = ["Rock", "Metal", "Hip-Hop", "Country", "Folk"]
    
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

def separate_stanzas_from_dataframe(data, n = 400):
    '''
    Splits lyrics up into datapoints that contain n words at a time
    
    Inputs:
    data: pandas DataFrame containing the data
    n (optional): the number of words per new datapoint. default is 400

    Outputs:
    separated_data: pandas DataFrame with separated stanzas
    '''

    if n is None:
        return data

    new_lyrics = []
    new_labels = []

    for i, row in data.iterrows():
        genre = row["Genre"]
        lyrics = row["Lyrics"]

        words = lyrics.split()
        for i in range(0, len(words), n):
            stanza = " ".join(words[i: i + n])

    separated_data = pd.DataFrame({"Genre": new_labels, "Lyrics": new_lyrics})

    return separated_data

def get_data(filename = "data.csv", clean_genre=True, 
             genres=None, num_included=None, 
             num_words_per_stanza = 400, training_ratio = 0.8,
             seed = 0):
    '''
    Input:
    filename: path to the .csv file name stored as a string
    clean_genre (optional): a boolean that is True if genres should be dropped
                            and the genre distribution should be uniform, and
                            is False otherwise
    genres (optional): list of accepted genres. default is None
    num_included (optional): number of each genre included. default is None
    num_words_per_stanza: the number of lines per new datapoint. Default is 4.
    training_ratio (optional): the proportion of data used for training, 
                               rest of data is for testing. Default is 0.8
    seed (optional): the random seed used. default is 0.

    Outputs:
    raw_data: the raw pandas dataframe with all data
    train_dict: a dictionary with keys {"lyrics": [...], "labels": [...]}
    test_dict: a dictionary with keys {"lyrics": [...], "labels": [...]}
    '''
    random.seed(seed)
    np.random.seed(seed)

    raw_data = load_raw_data(filename)
    cleaned_data = clean_data(raw_data)
    train_data, test_data = split_data(cleaned_data, genres, training_ratio)

    train_data = separate_stanzas_from_dataframe(train_data, num_words_per_stanza)
    test_data = separate_stanzas_from_dataframe(test_data, num_words_per_stanza)

    if clean_genre: 
        train_data = filter_genres(train_data, genres, num_included)
        test_data = filter_genres(test_data, genres, num_included)

    train_dict = dataframe_to_dict(train_data)
    test_dict = dataframe_to_dict(test_data)

    return raw_data, train_dict, test_dict

def get_information(data, lyrics):
    '''
    Gets information from data given the lyrics of a song (we assume that these are unique)

    Input:
    data: raw data
    lyrics: lyrics of the song we are searching for
    '''
    return data[data['Lyrics'].contains(lyrics)]

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

    
