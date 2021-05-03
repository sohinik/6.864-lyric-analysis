# 6.864-lyric-analysis
final project

## Dataset 
https://www.kaggle.com/mateibejan/multilingual-lyrics-for-genre-classification

## Data Cleaning
* Remove non-English songs
* Remove datapoints with missing data
* Keep only the genre and the lyrics 

After data cleaning, we have the following genre distribution

Rock          107145
Pop            86298
Metal          19133
Jazz           13314
Folk            8169
Indie           7240
R&B             2765
Hip-Hop         2238
Electronic      2005
Country         1890

We'll focus mainly on 
    Rock, Metal, Hip-Hop, Country, Folk 

## Featurization
* Transformers: 
    - tokenize: with embedding layer before transformer layer like in hw4
* RNNs: 
    - Word2Vec
    - tokenize: pass in token_ids to embedding layer then into RNN layer
    - tokenize: just token_ids into RNN layer