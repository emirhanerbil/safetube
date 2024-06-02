import boto3
from botocore.exceptions import NoCredentialsError
import json 

aws_access_key_id= ""
aws_secret_access_key = ""
bucket_name = ""
file_name = ""
filepath=""

def download_from_aws(bucket, s3_file, local_file):
    s3 = boto3.client('s3', 
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key)
    
    try:
        s3.download_file(bucket, s3_file, local_file)
        print("Download Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    
downloaded_model = download_from_aws(bucket_name, file_name, f"/home/ubuntu/{filepath}")
downloade_json = download_from_aws(bucket_name, file_name, f"/home/ubuntu/{filepath}")

print(downloaded_model,downloade_json)