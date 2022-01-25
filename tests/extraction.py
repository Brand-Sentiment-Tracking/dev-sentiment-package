import sparknlp

from brand_sentiment import ArticleExtraction

spark = sparknlp.start()
article_extractor = ArticleExtraction()
article = article_extractor.import_one_article('data/article.txt')
print(article)
sentences = article_extractor.article_to_sentences(article)
print(sentences)
