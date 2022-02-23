import pandas as pd
import numpy as np
import json
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline

from brand_sentiment.extraction import ArticleExtraction
from brand_sentiment.sentiment import SentimentIdentification
from brand_sentiment.identification import BrandIdentification


if __name__ == '__main__':
    spark = sparknlp.start()
    article_extractor = ArticleExtraction()
    brand_identifier = BrandIdentification()
    sentimentiser = BrandSentiment()
    article = article_extractor.import_one_article('data/article.txt')
    print(article)
