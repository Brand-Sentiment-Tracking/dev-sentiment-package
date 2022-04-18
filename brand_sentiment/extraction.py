from lib2to3.pgen2.pgen import DFAState
import nltk.data
import os
import json
import sparknlp


class ArticleExtraction:
    def __init__(self):
        self.dates_and_headlines = []

    def import_one_article(self, filepath):

        with open(filepath, "rb") as f:
            article = f.read().decode('utf-8')

        return article

    def import_one_headline_json(self, filepath):
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
                # Store the date and headline as a list of tuples
                self.dates_and_headlines.append((data['date_download'], data['title'])) 
            except:
                print('SKIP')

    def article_to_sentences(self, article):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        return '\n-----\n'.join(tokenizer.tokenize(article))

    def import_subfolder_headlines(self, folderpath):
        for filename in os.listdir(folderpath):
            filepath = os.path.join(folderpath, filename)
            self.import_one_headline_json(filepath)
        return self.headlines

    def import_folder_headlines(self, folderpath):
        for root, dirs, files in os.walk(folderpath):
            for name in files:
                self.import_one_headline_json(os.path.join(root, name))

        spark = sparknlp.start()
        # The input spark df must have a "text" column to pass to model
        input_df = spark.createDataFrame(self.dates_and_headlines, ["date_download", "text"]) 

        return input_df


if __name__ == '__main__':
    article_extractor = ArticleExtraction()
    # article_extractor.import_subfolder_headlines('./data/cc_download_articles/cyprus-mail.com')
    df = article_extractor.import_folder_headlines('./data/cc_download_articles')
    print((df.count(), len(df.columns)))
