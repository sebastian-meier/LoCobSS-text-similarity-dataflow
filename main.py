import mysql.connector

from dotenv import load_dotenv
load_dotenv()

import os

from google.cloud import storage

import tensorflow_hub as hub
import numpy as np

from sklearn import manifold
from sklearn.metrics import pairwise_distances
from sklearn import decomposition

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main(args):

  # Some of the multidimensional-scaling approaches are really slow, especially without GPU
  # We got the best results from tsne and its quite fast as well, therefore all other approaches
  # are disabled by default. Set True, to run all approaches.
  run_all = False

  # ----------------------
  # Connect to questions-database and save all questions to a text file (questions.txt)

  db = mysql.connector.connect(
    host=os.environ.get('MYSQL_SERVER'),
    user=os.environ.get('MYSQL_USER'),
    password=os.environ.get('MYSQL_PASS'),
    database=os.environ.get('MYSQL_DB')
  )

  cursor = db.cursor()

  cursor.execute("SELECT id, question_en FROM questions")

  result_q = []
  result_id = []

  for r in cursor.fetchall():
    result_id.append(str(r[0]))
    if r[1] == None:
      result_q.append("")
    else:
      result_q.append(r[1].decode("utf-8"))
  
  cursor.close()
  db.close()

  print('db update ✅')

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

  print('embedding ✅')

  def map_2d_array(a):
    for axis in range(2):
      max = 0
      min = 0
      for i, row in enumerate(a):
        if i == 0:
          max = row[axis]
          min = row[axis]
        else:
          if max < row[axis]:
            max = row[axis]
          if min > row[axis]:
            min = row[axis]
      for i, row in enumerate(a):
        a[i][axis] = (row[axis] - min) / (max - min)
    return a

  if run_all:
    # distance matrix to 2d coords using MDS
    mds_model = manifold.MDS(n_components=2, dissimilarity='precomputed', random_state=1, n_init=4, max_iter=100, n_jobs=-1)
    distances = pairwise_distances(data)
    mds_out = map_2d_array(mds_model.fit_transform(distances))
    np.savetxt('/tmp/mds.gz', np.array(mds_out), delimiter=",", fmt='%.4f')
    blob = bucket.blob(os.environ.get('GS_FILE_MDS'))
    blob.content_encoding = 'gzip'
    blob.content_type = 'text/csv'
    blob.upload_from_filename('/tmp/mds.gz')

    print('mds ✅')

  # distance matrix to 2d coords using t-SNE
  tsne_model = manifold.TSNE(n_components=2, init='pca', random_state=1, n_jobs=-1)
  tsne_out = map_2d_array(tsne_model.fit_transform(embeddings))
  np.savetxt("/tmp/tsne.gz", np.array(tsne_out), delimiter=",", fmt='%.4f')
  blob = bucket.blob(os.environ.get('GS_FILE_TSNE'))
  blob.content_encoding = 'gzip'
  blob.content_type = 'text/csv'
  blob.upload_from_filename('/tmp/tsne.gz')

  print('tsne ✅')

  if run_all:
    # distance matrix to 2d coords using PCA
    pca_model = decomposition.PCA(n_components=2, random_state=1)
    pca_out = map_2d_array(pca_model.fit_transform(embeddings))
    np.savetxt("/tmp/pca.gz", np.array(pca_out), delimiter=",", fmt='%.4f')
    blob = bucket.blob(os.environ.get('GS_FILE_PCA'))
    blob.content_encoding = 'gzip'
    blob.content_type = 'text/csv'
    blob.upload_from_filename('/tmp/pca.gz')

    print('pca ✅')

  db = mysql.connector.connect(
    host=os.environ.get('MYSQL_SERVER'),
    user=os.environ.get('MYSQL_USER'),
    password=os.environ.get('MYSQL_PASS'),
    database=os.environ.get('MYSQL_DB')
  )

  cursor = db.cursor()

  extra_columns = ""
  if run_all:
    extra_columns = ", pca_x, pca_y, mds_x, mds_y"

  query = "INSERT INTO questions_coordinates (id, tsne_x, tsne_y%s) VALUES ".format(extra_columns)
  for i, row in enumerate(result_id):
    if i > 0:
      query += ','
    query += "({},{:.4f},{:.4f}".format(result_id[i], tsne_out[i][0], tsne_out[i][1])
    if run_all:
      query += ",{:.4f},{:.4f},{:.4f},{:.4f}".format(pca_out[i][0], pca_out[i][1], mds_out[i][0], mds_out[i][1])
    query += ")"

  cursor.execute("TRUNCATE TABLE questions_coordinates")
  cursor.execute(query)
  cursor.execute("""UPDATE questions INNER JOIN questions_coordinates ON questions.id = questions_coordinates.id
    SET questions.tsne_x = questions_coordinates.tsne_x,
    questions.tsne_y = questions_coordinates.tsne_y,
    questions.pca_x = questions_coordinates.pca_x,
    questions.pca_y = questions_coordinates.pca_y,
    questions.mds_x = questions_coordinates.mds_x,
    questions.mds_y = questions_coordinates.mds_y""")

  cursor.close()
  db.close()

  print('db updated ✅')

  return 'embeddings up to date'

def demo():
  print("demo")
