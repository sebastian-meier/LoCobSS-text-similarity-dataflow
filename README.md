# LoCobSS-text-similarity-dataflow
Collecting text strings and embeddings from a MYSQL database, caching them as text files on Google Cloud Storage. The resulting embeddings are used for similarity search: [LoCobSS-text-similarity](https://github.com/sebastian-meier/LoCobSS-text-similarity).

## Configuration
Create a **.env** file based on the **.env-sample**.

## Local testing
You need to add *GOOGLE_APPLICATION_CREDENTIALS* to the **.env** file with the path to your credentials file.
```bash
pip install -r requirements.txt
python test.py
```

## Deploy
```bash
gcloud functions deploy questions_mysql-to-cloud-storage --entry-point main --runtime python38 --trigger-http --region europe-west3 --memory 4G
```

## Additional scripts

### batch embeddings

`./batch/main.py`

This gets all current questions from the database, creates embeddings and uploads the resulting vectors to the database. The script only collects those questions that do not already have a set of vectors. Clear the vector table, if you want all vectors to be (re-)created.