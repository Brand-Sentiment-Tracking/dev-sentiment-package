import nltk.data


class ArticleExtraction:
    def __init__(self):
        pass

    def import_one_article(self, filepath):

        with open(filepath, "rb") as f:
            article = f.read().decode('utf-8')

        return article

    def article_to_sentences(self, article):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        return '\n-----\n'.join(tokenizer.tokenize(article))
