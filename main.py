import mysql.connector

from dotenv import load_dotenv
load_dotenv()

import os

from google.cloud import storage

import tensorflow_hub as hub
import numpy as np

def main(args):
  # ----------------------
  # Connect to questions-database and save all questions to a text file (questions.txt)

  db = mysql.connector.connect(
    host=os.environ.get('MYSQL_SERVER'),
    user=os.environ.get('MYSQL_USER'),
    password=os.environ.get('MYSQL_PASS'),
    database=os.environ.get('MYSQL_DB')
  )

  cursor = db.cursor()

  cursor.execute("SELECT id, question FROM {}".format(os.environ.get('MYSQL_TABLE')))

  result_q = []
  result_id = []

  for r in cursor.fetchall():
    result_id.append(str(r[0]))
    result_q.append(r[1].decode("utf-8"))
  
  storage_client = storage.Client()
  bucket = storage_client.bucket(os.environ.get('GS_BUCKET'))

  # store question and ids as text files
  blob = bucket.blob(os.environ.get('GS_FILE_IDS'))
  blob.upload_from_string('\n'.join(result_id))

  blob = bucket.blob(os.environ.get('GS_FILE_QS'))
  blob.upload_from_string('\n'.join(result_q))

  # embed questions
  module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
  model = hub.load(module_url)
  embeddings = np.array(model(result_q))

  # store embeddings
  data = np.array(embeddings)
  np.save('/tmp/temp.npy', data)

  blob = bucket.blob(os.environ.get('GS_FILE_NPY'))
  blob.upload_from_filename('/tmp/temp.npy')

  return 'embeddings up to date'

def demo():
  print("demo")
