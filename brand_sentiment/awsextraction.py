import os
import json
import boto3

# all_files = boto3.list_files()
# --> ["2022-06-03/www_bbc_co_uk_news_something.json", ...]
# for each file in all_files:
#     path_list = os.path.split(file)
# --> ["2022-06-03", "www_bbc_co_uk_news_something.json"]
# earlest_date = min(["2022-06-03", "2022-06-02"])
# all_files_to_process = glob(f"{earliest_date}/*", all_files)
# for each file in all_files_to_process:
#     fp = boto3.get_file(file)
#
# objs = boto3.client.list_objects(Bucket='extractedarticlesdev')
# response = s3.list_objects(Bucket=🗑)
# all_files_to_process = glob.glob(f"{earliest_date}/*", self.article_paths)


class ArticleExtraction:
    def __init__(self):
        self.headlines = []
        self.s3 = boto3.client("s3",
                               region_name='eu-west-2',
                               aws_access_key_id=ACCESS_KEY,
                               aws_secret_access_key=SECRET_ACCESS_KEY)
        # print(self.s3.list_buckets())
        self.bucket = 'extracted-articles-dev'
        # self.article_paths = self.s3.list_objects(Bucket=self.bucket)
        # print(self.article_paths)

    def _create_path_dict(self):
        response = self.s3.list_objects(Bucket=self.bucket)
        request_files = response["Contents"]
        path_dict = {}
        for file in request_files:
            filepath = file['Key']
            date = os.path.split(filepath)[0]
            path_dict.setdefault(date, []).append(filepath)
        return path_dict

    def _find_earliest_date(self, path_dict):
        earliest_date = min(set(path_dict.keys()))
        return str(earliest_date)

    # def read_earliest_articles(self):
    #     earliest_date = self.find_earliest_date()
    #     for article in self.path_dict:
    #         obj = self.s3.get_object(Bucket=self.bucket, Key=file[earliest_date])

    def read_earliest_articles(self):
        path_dict = self._create_path_dict()
        earliest_date = self._find_earliest_date(path_dict)
        headlines = []
        for filepath in path_dict[earliest_date]:
            obj = self.s3.get_object(Bucket=self.bucket, Key=filepath)
            article = obj['Body'].read().decode("utf-8")

            with open('test.json', 'w') as f:
                f.write(article)
            try:
                article = json.loads(article)
                headlines.append(article['title'])
            except json.decoder.JSONDecodeError:
                print(article)
        return headlines


if __name__ == '__main__':
    article_extractor = ArticleExtraction()
    print(article_extractor.read_earliest_articles())
