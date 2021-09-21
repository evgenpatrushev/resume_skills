# import libraries
import re
import nltk
from nltk.corpus import stopwords

en_stops = set(stopwords.words('english'))
for i in ['sir', 'sr']:
    en_stops.add(i)

import pandas as pd
import matplotlib.pyplot as plt


# function for cleaning the data
# get resume as input, delete html tags, https urls, etc and divide resume by sentences

# Return list of lists. Each list is generated sentence. These lists contains lowcase words (without stopwords)
def clean_text(raw_data):
    raw_data = raw_data.lower()
    clean_html = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = re.sub(clean_html, '', raw_data)

    text = re.sub('\r\n', '. ', text)

    text = re.sub(r'http\S+', '', text)

    # remove any numeric characters
    text = ''.join([word for word in text if not word.isdigit()])
    # remove RT and cc
    text = re.sub('RT|cc', ' ', text)
    # remove hashtags
    text = re.sub(r'#\S+', '', text)
    # remove mentions
    text = re.sub(r'@\S+', '  ', text)
    # replace consecutive non-ASCII characters with a space
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    # extra whitespace removal
    text = re.sub(r'\s+', ' ', text)

    #     text = text.replace('  ', '. ')
    text = text.split('. ')

    text = [i.strip() for i in text if len(i) > 1]

    for i, sentns in enumerate(text):
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(sentns)
        words = [w for w in words if w not in en_stops]
        text[i] = words

    while True:
        clear = True
        index = []
        for i, sent in enumerate(text):
            if len(sent) <= 2 and i != len(text) - 1:
                clear = False
                text[i + 1] = text[i] + text[i + 1]
                text.pop(i)
            elif len(sent) <= 2 and i == len(text) - 1:
                clear = False
                text[i - 1] = text[i - 1] + text[i]
                text.pop(i)
        if clear:
            break

    return text


def dataset_preprocess(df, columns_to_use=None, plotting=False):

    # For training the model with this data set will use more or less balanced categories, which are listed below.
    if columns_to_use is None:
        columns_to_use = ['DevOps Engineer', 'Hadoop', 'Database', 'Data Science', 'Python Developer']

    df = df.drop_duplicates(ignore_index=True)

    # plotting ------------------------------------------------

    if plotting:
        pd.value_counts(df['Category']).plot.bar(figsize=(15, 5))
        plt.title('Distribution of "job titles" in dataset')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.show()

        df['Resume_length'] = df['Resume'].str.len()

        resume_length = df.groupby('Category').mean().sort_values(by='Resume_length', ascending=False)['Resume_length']
        plt.figure(figsize=(15, 5))
        plt.plot(resume_length)
        plt.xticks(rotation=90)
        plt.show()

    # ---------------------------------------------------------
    df_sliced = df[df['Category'].isin(columns_to_use)]
    df_sliced = df_sliced.reset_index(drop=True)
    df_sliced['Category'].unique()

    # apply this function to resumes in DataFrame
    df_sliced['Resume'] = df_sliced['Resume'].apply(clean_text)

    return df_sliced


if __name__ == '__main__':
    # load dataset. Dataset from kaggle https://www.kaggle.com/gauravduttakiit/resume-dataset
    dataframe = pd.read_csv('UpdatedResumeDataSet.csv')
    dataset_preprocess(dataframe)
