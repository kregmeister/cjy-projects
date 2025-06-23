#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 22:25:52 2024

@author: cjymain
"""

import boto3
import json
from time import sleep

from technically.utils.log import get_logger


def get_credentials(keys: list):
    "Retreives sensitive credentials from a secure location (AWS Secrets Manager)."

    retries = 0
    while True:
        secret_name = "TechnicallyAuth"
        region_name = "us-east-1"

        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )

        try:
            get_secret_value_response = client.get_secret_value(
                SecretId=secret_name
            )
        except Exception as e:
            get_logger().error(f"Error connecting to AWS Secrets Manager: {e}. Retrying... ({retries})")
            if retries == 10:
                raise e
            retries += 1
            sleep(retries ** 2)
            continue

        secret = json.loads(get_secret_value_response["SecretString"])
        
        resp = [value for key, value in secret.items() if key in keys]
        
        if len(resp) == 1:
            return resp[0]
        else:
            return resp


        