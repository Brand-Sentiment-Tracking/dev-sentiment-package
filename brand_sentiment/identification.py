from brand_sentiment.extraction import ArticleExtraction

import os
import sparknlp
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import sparknlp
from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.pretrained import PretrainedPipeline
from sparknlp_display import NerVisualizer


class BrandIdentification:
    def __init__(self, MODEL_NAME):
        self.MODEL_NAME = MODEL_NAME

        # Define Spark NLP pipeline
        documentAssembler = DocumentAssembler() \
            .setInputCol('text') \
            .setOutputCol('document')

        tokenizer = Tokenizer() \
            .setInputCols(['document']) \
            .setOutputCol('token')

        # ner_dl and onto_100 model are trained with glove_100d, so the embeddings in the pipeline should match
        if (self.MODEL_NAME == "ner_dl") or (self.MODEL_NAME == "onto_100"):
            embeddings = WordEmbeddingsModel.pretrained('glove_100d') \
                .setInputCols(["document", 'token']) \
                .setOutputCol("embeddings")

        # Bert model uses Bert embeddings
        elif self.MODEL_NAME == "ner_dl_bert":
            embeddings = BertEmbeddings.pretrained(name='bert_base_cased', lang='en') \
                .setInputCols(['document', 'token']) \
                .setOutputCol('embeddings')

        ner_model = NerDLModel.pretrained(MODEL_NAME, 'en') \
            .setInputCols(['document', 'token', 'embeddings']) \
            .setOutputCol('ner')

        ner_converter = NerConverter() \
            .setInputCols(['document', 'token', 'ner']) \
            .setOutputCol('ner_chunk')

        nlp_pipeline = Pipeline(stages=[
            documentAssembler,
            tokenizer,
            embeddings,
            ner_model,
            ner_converter
        ])

        # Create the pipeline model
        empty_df = spark.createDataFrame([['']]).toDF('text')
        self.pipeline_model = nlp_pipeline.fit(empty_df)


    def create_ranked_result_df(self, text):
        # Run the pipeline for the text
        text_df = spark.createDataFrame(pd.DataFrame({'text': text}, index = [0]))
        result = self.pipeline_model.transform(text_df)

        # Tabulate results
        df = result.select(F.explode(F.arrays_zip('document.result', 'ner_chunk.result',"ner_chunk.metadata")).alias("cols")).select(\
        F.expr("cols['1']").alias("chunk"),
        F.expr("cols['2'].entity").alias('result'))

        # Rank the identified ORGs by frequencies
        ranked_df = df.filter(df.result == 'ORG').groupBy(df.chunk).count().orderBy('count', ascending=False)

        return ranked_df


    def predict_by_headline(self, headline):
        ranked_df_hl = self.create_ranked_result_df(headline)
        # Print the table
        ranked_df_hl.show(100, truncate=False)

        # If only one ORG appears in headline, return it
        # If an ORG appears more than the others, return it
        if ranked_df_hl.count() == 1:
        # if ranked_df_hl.count() == 1 or ranked_df_hl.first()[1] > ranked_df_hl.collect()[1][1]:
            return ranked_df_hl.first()[0]
        else: # If no ORG appears, or multiple ORGs appear the same time, return None
            return None


    def predict(self, body):
        ranked_df = self.create_ranked_result_df(body)
        # Print the table
        ranked_df.show(100, truncate=False)

        # Return the ORG with highest freq if the freq >= 2
        if ranked_df.first()[1] >= 2:
            return ranked_df.first()[0]
        else:
            return None
        # TO DO: break even - Wikidata#



if __name__ == '__main__':
    article_extractor = ArticleExtraction()
    script_dir = os.path.dirname(__file__)  # Script directory
    headline_full_path = os.path.join(script_dir, 'data/headline.txt')
    body_full_path = os.path.join(script_dir, 'data/article_body.txt')

    spark = sparknlp.start()

    MODEL_NAME = "ner_dl_bert"
    # MODEL_NAME = "onto_100"
    brand_identifier = BrandIdentification(MODEL_NAME)

    headline = article_extractor.import_one_article(headline_full_path)
    body = article_extractor.import_one_article(body_full_path)

    brand_by_headline = brand_identifier.predict_by_headline(headline)
    print(headline)
    print(brand_by_headline)

    # Only use article body if no brand is identified in the headline
    if brand_by_headline == None:
        brand = brand_identifier.predict(body)
        print(brand)
