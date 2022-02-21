from brand_sentiment.extraction import ArticleExtraction

import sparknlp
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
# from tabulate import tabulate
from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.pretrained import PretrainedPipeline


class SentimentIdentification:
    def __init__(self, MODEL_NAME):
        """Creates a class for sentiment identication using specified model.

        Args:
          MODEL_NAME: Name of the Spark NLP pretrained pipeline.
        """

        # Create the pipeline instance
        self.MODEL_NAME = MODEL_NAME

        if self.MODEL_NAME == "custom_pipeline": # https://nlp.johnsnowlabs.com/2021/11/03/bert_sequence_classifier_finbert_en.html
          document_assembler = DocumentAssembler() \
              .setInputCol('text') \
              .setOutputCol('document')

          tokenizer = Tokenizer() \
              .setInputCols(['document']) \
              .setOutputCol('token')

          sequenceClassifier = BertForSequenceClassification \
                .pretrained('bert_sequence_classifier_finbert', 'en') \
                .setInputCols(['token', 'document']) \
                .setOutputCol('class') \
                .setCaseSensitive(True) \
                .setMaxSentenceLength(512)

          pipeline = Pipeline(stages=[
              document_assembler,
              tokenizer,
              sequenceClassifier
          ])

          self.pipeline_model = LightPipeline(pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

        else:
          self.pipeline_model = PretrainedPipeline(self.MODEL_NAME, lang = 'en')



    def predict(self, text):
        """Predicts sentiment of the input string..

        Args:
          text: String to classify.
        """
        self.text = text

        # Annotate input text using pretrained model
        annotations =  self.pipeline_model.annotate(self.text)

        # Depending on the chosen pipeline the outputs will be slightly different
        if self.MODEL_NAME == "analyze_sentimentdl_glove_imdb":
          # print(f"{annotations['sentiment']} {annotations['document']}")

          if isinstance(self.text, list):
            return [annotation['sentiment'][0] for annotation in annotations] # Return the sentiment list of strings
          else:
            return annotations['sentiment'][0] # Return the sentiment string

        else:
          # print(f"{annotations['class']} {annotations['document']}")

          if isinstance(self.text, list):
            return [annotation['class'][0] for annotation in annotations] # Return the sentiment list of strings
          else:
            return annotations['class'][0] # Return the sentiment string


if __name__ == '__main__':
    spark = sparknlp.start()
    
    MODEL_NAME = "ner_dl_bert"
    # MODEL_NAME = "onto_100"
    article_extractor = ArticleExtraction
    article = article_extractor.import_one_article('data/article.txt')

    # identifier = SentimentIdentification(MODEL_NAME =  "analyze_sentimentdl_glove_imdb")
    # identifier = SentimentIdentification(MODEL_NAME =  "classifierdl_bertwiki_finance_sentiment_pipeline")
    identifier = SentimentIdentification(MODEL_NAME="custom_pipeline")  # Uses https://nlp.johnsnowlabs.com/2021/11/03/bert_sequence_classifier_finbert_en.html

    # Predict by headline
    headline = article[0]
    identifier.predict(headline)

    # Predict by body
    body = article[1]
    identifier.predict(body)


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
