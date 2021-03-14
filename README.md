# LoCobSS-text-similarity-dataflow
Collecting text strings froma MYSQL database, caching them as text files on Google Cloud Storage, creating embeddings from the text elements through [universal-sentence-encoder](https://tfhub.dev/google/universal-sentence-encoder/4) and storing the resulting embeddings also on Google Cloud Storage. The resulting embeddings are used for similarity search: [LoCobSS-text-similarity](https://github.com/sebastian-meier/LoCobSS-text-similarity).

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

## Legacy
We started exploring Google Dataflow, but found it too much hassle for something so straight forward.