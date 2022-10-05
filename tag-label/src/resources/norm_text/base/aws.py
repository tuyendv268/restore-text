
import os
import boto3

class AWS():
    def __init__(self, aws_access_key_id, aws_secret_access_key, bucket):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.bucket = bucket
        self.s3 = boto3.client('s3', aws_access_key_id=self.aws_access_key_id, aws_secret_access_key=self.aws_secret_access_key)
        
    def upload(self, local_file_path, s3_path):
        self.s3.upload_file(local_file_path, self.bucket, s3_path)

    def download(self, s3_path, local_file_path):
        self.s3.download_file(self.bucket, s3_path, local_file_path)