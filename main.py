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
from IPython.display import display
import time

if __name__ == '__main__':
    spark = sparknlp.start()
    article_extractor = ArticleExtraction()
    brand_identifier = BrandIdentification("ner_dl_bert")
    sentimentiser = SentimentIdentification(MODEL_NAME="classifierdl_bertwiki_finance_sentiment_pipeline")
    # article = article_extractor.import_one_article('data/article.txt')
    # print(article)
    # print(test())
    headlines = article_extractor.import_folder_headlines('data/cc_download_articles/cyprus-mail.com')
    print(headlines)

    # Measure prediction speed
    start = time.time()

    brand_spark_df = brand_identifier.predict_brand(headlines)
    complete_spark_df = sentimentiser.predict_dataframe(brand_spark_df)
    complete_spark_df.show()

    end = time.time()
    print(f"Time to identify brands and predict sentiment {end-start}")

    # Display as pandas dataframe for better visualization
    display(complete_spark_df.toPandas()) 