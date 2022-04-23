from google.cloud import bigquery
from google.cloud import storage
import logging
import os
import csv 

# Set up logging
logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s',
                     datefmt='%Y-%m-%d %H:%M:%S')

PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT') 
logging.info('Google Cloud project is {}'.format(PROJECT))

BUCKET_NAME = PROJECT + '.appspot.com'
logging.info('Initialising access to storage bucket {}'.format(BUCKET_NAME))
APP_BUCKET = storage.Client().bucket(BUCKET_NAME)

logging.info('Initialising BigQuery client')
BQ_CLIENT = bigquery.Client()

def get_images(description,writer):
    results = BQ_CLIENT.query(
    ''' 
        Select ImageID 
        FROM bdcc-proj.openimages.image_labels
        JOIN bdcc-proj.openimages.classes USING(Label)
        Where Description = "{0}"
        LIMIT 100
    '''.format(description)).result()
    count = 0
    imgtype = "TRAIN"
    for result in results:
        if count > 79: 
            imgtype = "TEST"
        if count > 89:
            imgtype = "VALIDATION"

        image = "gs://bdcc-proj-images/bdcc-proj-vcm/img/images/" + result[0] + ".jpg"
        line = [imgtype,image,description]
        writer.writerow(line)
        count = count + 1

if __name__ == '__main__':
    descriptions = ["Antelope","Beetle","Butterfly","Cat","Chicken","Dog","Dolphin","Elephant","Goat","Goose","Hamster","Hedgehog","Horse","Monkey","Parrot"]
    with open('/home/maramadeu/bdcc-proj/automl.csv', 'w') as f: 
        writer = csv.writer(f)
        for desc in descriptions: 
            get_images(desc,writer)
    
