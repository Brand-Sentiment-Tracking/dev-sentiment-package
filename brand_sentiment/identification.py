from extraction import ArticleExtraction

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
        """ Sets up pipeline.
        """
        self.MODEL_NAME = MODEL_NAME

        # Spark NLP stage to convert data stored in DataFrame
        # default is inputting text column and adding document column
        documentAssembler = DocumentAssembler() \
            .setInputCol('text') \
            .setOutputCol('document')

        # Spark NLP stage to ???
        tokenizer = Tokenizer() \
            .setInputCols(['document']) \
            .setOutputCol('token')

        # Spark NLP stages for Word Embeddings
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

        # Model to detect
        ner_model = NerDLModel.pretrained(MODEL_NAME, 'en') \
            .setInputCols(['document', 'token', 'embeddings']) \
            .setOutputCol('ner')

        ner_converter = NerConverter() \
            .setInputCols(['document', 'token', 'ner']) \
            .setOutputCol('ner_chunk')

        # Spark NLP Pipeline
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
        ranked_df_hl.show(100, truncate=False)

        # If only one ORG appears in headline, return it
        if ranked_df_hl.count() == 1:
            return ranked_df_hl.first()[0]
        else:  # If no ORG appears, or multiple ORGs all appear once, return no brand
            return None

    def predict(self, body):
        ranked_df = self.create_ranked_result_df(body)
        ranked_df.show(100, truncate=False)

        # Return the ORG with highest freq (at least greater than 2)
        if ranked_df.first()[1] > 2:
            return ranked_df.first()[0]
        else:
            return None
        # TO DO: break even - Wikidata#

    # def visualise(self, ranked_df, result):
        # Visualise ORG names in text
        # NerVisualizer().display(
            # result = result.collect()[0],
            # label_col = 'ner_chunk',
            # document_col = 'document',
            # labels=['ORG']
    #         )


if __name__ == '__main__':

    MODEL_NAME = "ner_dl_bert"
    # MODEL_NAME = "onto_100"
    article_extractor = ArticleExtraction()
    article = article_extractor.import_one_article("./data/article.txt")

    spark = sparknlp.start()
    brand_identifier = BrandIdentification(MODEL_NAME)
    headline, body = article

    headline_brand = brand_identifier.predict_by_headline(headline)
    print(headline)
    print(headline_brand)

    # Only use article body if no brand identified in the headline
    if headline_brand is None:
        brand = brand_identifier.predict(body)
        print(brand)
