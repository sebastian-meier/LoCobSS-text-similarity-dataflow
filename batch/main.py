import mysql.connector

from dotenv import load_dotenv
load_dotenv()

import os

import tensorflow_hub as hub
import numpy as np

db = mysql.connector.connect(
  host=os.environ.get('MYSQL_SERVER'),
  user=os.environ.get('MYSQL_USER'),
  password=os.environ.get('MYSQL_PASS'),
  database=os.environ.get('MYSQL_DB')
)

cursor = db.cursor()

cursor.execute("""SELECT 
  id,
  question_en,
  question_id
FROM
  questions
LEFT OUTER JOIN
  question_vectors
  ON question_vectors.question_id = questions.id
WHERE
  question_id IS NULL""")

result_q = []
result_id = []

for r in cursor.fetchall():
  result_id.append(str(r[0]))
  if r[1] == None:
    result_q.append("")
  else:
    result_q.append(r[1].decode("utf-8"))

# depending on db size, the embedding can take a while and the db-connection might close. so we close it and re-open it later.
cursor.close()
db.close()

print('db select ✅')
if len(result_q) > 0:

  # embed questions
  module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
  model = hub.load(module_url)
  embeddings = np.array(model(result_q))

  # store embeddings
  data = np.array(embeddings)

  print('embeddings ✅')

  db = mysql.connector.connect(
    host=os.environ.get('MYSQL_SERVER'),
    user=os.environ.get('MYSQL_USER'),
    password=os.environ.get('MYSQL_PASS'),
    database=os.environ.get('MYSQL_DB')
  )

  cursor = db.cursor()

  extra_columns = ""
  for i in range(512):
    extra_columns += ", vec_{}".format(i + 1)

  insert_query = "INSERT INTO question_vectors (question_id{}) VALUES ".format(extra_columns)

  queries = []
  query_limit = 500
  query_current = 0

  query = ""
  for i, row in enumerate(embeddings):
    if query_current > 0:
      query += ','
    query += "({}".format(result_id[i])
    for j, col in enumerate(row):
      query += ",{}".format(col)
    query += ")"
    query_current += 1
    if query_current > query_limit:
      query_current = 0
      queries.append(query)
      query = ""
  queries.append(query)

  for q in queries:
    cursor.execute(insert_query + q)

  cursor.close()
  db.close()

  print('db updated ✅')

else:

  print('no questions to update')
