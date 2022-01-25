import sparknlp
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from tabulate import tabulate
import sparknlp
from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.pretrained import PretrainedPipeline


class BrandSentiment:
    def __init__(self, model_name):
        self.model_pipeline = PretrainedPipeline('analyze_sentimentdl_glove_imdb', lang='en')

    def one_sentence_sentiment(self, sentence):
        """Sentiment of one sentence using pretrained model given to class.

        Args:
            sentence (str): Sentence to do sentiment analysis on.

        Returns:
            str: The return value. True for success, False otherwise.

        """
        return self.model_pipeline.annotate(sentence)

    def multiple_sentences_sentiments(self, sentences):
        return [self.one_sentence_sentiment(sentence) for sentence in sentences]

    def aggregate_sentences_sentiments(self, sentiments):
        total_sentiment = 0
        for sentiment in sentiments:
            if sentiment == 'pos':
                total_sentiment += 1
            elif sentiment == 'neg':
                total_sentiment -= 1
        return total_sentiment/len(sentiments)

    def analyse_article(self, sentences):
        sentiments = self.multiple_sentences_sentiments(sentences)
        return self.aggregate_sentences_sentiments(sentiments)
