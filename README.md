# BOOLEAN INFORMATION RETRIEVAL SYSTEM

Data Preprocessing:
Removal of stopwords
Stemming
Lemmatization
WildCard Query Handling
Spelling correction: Edit distance method

Inverted Index
Query Management

> Part of Information Retrieval 

## Information Retrieval

### Dataset

The dataset used for this purpose is taken from class files of Shakespeares Literature docs.

### Usage

Run the Code,
Enter the Query and find the results of a Boolean Information Retrieval using Inverted Index

### Test Case:

## Example1
```
PS C:\Users\shrey> python -u "c:\Projects\Information Retrieval\test1.py"
Enter your query:julius
Connecting words: []
julius
[[0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
['antony-and-cleopatra_TXT_FolgerShakespeare.txt', 'cymbeline_TXT_FolgerShakespeare.txt', 'hamlet_TXT_FolgerShakespeare.txt', 'henry-vi-part-1_TXT_FolgerShakespeare.txt', 'henry-vi-part-2_TXT_FolgerShakespeare.txt', 'julius-caesar_TXT_FolgerShakespeare.txt', 'richard-iii_TXT_FolgerShakespeare.txt', 'richard-ii_TXT_FolgerShakespeare.txt']
```


## Example2
```
PS C:\Users\shrey> python -u "c:\Projects\Information Retrieval\test1.py"
Enter your query:julius and king
Connecting words: ['and']
julius
king
[[0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]]
[[0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
['antony-and-cleopatra_TXT_FolgerShakespeare.txt', 'cymbeline_TXT_FolgerShakespeare.txt', 'hamlet_TXT_FolgerShakespeare.txt', 'henry-vi-part-1_TXT_FolgerShakespeare.txt', 'henry-vi-part-2_TXT_FolgerShakespeare.txt', 'julius-caesar_TXT_FolgerShakespeare.txt', 'richard-iii_TXT_FolgerShakespeare.txt', 'richard-ii_TXT_FolgerShakespeare.txt']
```