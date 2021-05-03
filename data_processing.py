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

def clean_data(data, clean_genre=True):
    '''
    Input:
    data: a pandas DataFrame storing the data in 'train.csv
    clean_genre: a boolean that is True if genres should be dropped
                 and the genre distribution should be uniform, and
                 is False otherwise

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

    if clean_genre: cleaned_data = genre_cleaner(cleaned_data)

    return cleaned_data

def genre_cleaner(data, genres=None, num_included=110000):
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
    num_included (optional): number of each genre included. default is 110000

    Output:
    cleaned_genre_data: cleaned version of data with max num_included points of the given genres
    '''
    if not genres:
        genres = ["Rock", "Metal", "Hip-Hop", "Country", "Folk"]

    cleaned_genre_data = None

    for genre in genres:
        sampled_genre_data = data[data["Genre"] == genre].sample(n = num_included)
        if cleaned_genre_data:
            cleaned_genre_data = sampled_genre_data
        else:
            cleaned_genre_data = cleaned_genre_data.append(sampled_genre_data)
    
    return cleaned_genre_data

def split_data(data, training_ratio = 0.8):
    '''
    Randomly splits all of the data into either the training
    or testing data set

    Input:
    data: a pandas DataFrame containing all of the data
    training_ratio (optional): the proportion of data used for training, 
                               rest of data is for testing. Default is 0.8

    Outputs:
    train_data: a pandas DataFrame containing the training_data
    test_data: a pandas DataFrame containing the testing_data
    
    '''
    num_datapoints = len(data.index)
    indices = np.arange(num_datapoints)
    np.random.shuffle(indices)

    num_training = training_ratio * num_datapoints
    training_indices = indices[ :num_training]
    testing_indices = indices[num_training: ]

    train_data = data.iloc[training_indices]
    test_data = data.iloc[testing_indices]

    return train_data, test_data

def get_data(filename = "data.csv"):
    '''
    Input:
    filename: path to the .csv file name stored as a string

    Outputs:
    train_data: a pandas DataFrame containing the training_data
    test_data: a pandas DataFrame containing the testing_data
    '''
    data = load_raw_data(filename)
    data = clean_data(data)
    train_data, test_data = split_data(data)

    return train_data, test_data

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

    
