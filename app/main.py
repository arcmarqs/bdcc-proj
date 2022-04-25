# Imports
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import flask
import logging
import os
import tfmodel
from google.cloud import bigquery
from google.cloud import storage

# Set up logging
logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s',
                     datefmt='%Y-%m-%d %H:%M:%S')

PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT') 
KEY = os.environ.setdefault('GOOGLE_CLOUD_PROJECT',"/home/maramadeu/bdcc-proj-762bedfa280a.json") 
logging.info('Google Cloud project is {}'.format(PROJECT))

# Initialisation
logging.info('Initialising app')
app = flask.Flask(__name__)

logging.info('Initialising BigQuery client')
BQ_CLIENT = bigquery.Client()

BUCKET_NAME = PROJECT + ".appspot.com"
logging.info('Initialising access to storage bucket {}'.format(BUCKET_NAME))
APP_BUCKET = storage.Client().bucket(BUCKET_NAME)

logging.info('Initialising TensorFlow classifier')
TF_CLASSIFIER = tfmodel.Model(
    app.root_path + "/static/tflite/my_model.tflite",
    app.root_path + "/static/tflite/my_dict.txt"
)
logging.info('Initialisation complete')

# End-point implementation
@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/classes')
def classes():
    results = BQ_CLIENT.query(
    '''
        Select Description, COUNT(*) AS NumImages
        FROM `bdcc-proj.openimages.image_labels`
        JOIN `bdcc-proj.openimages.classes` USING(Label)
        GROUP BY Description
        ORDER BY Description
    ''').result()
    logging.info('classes: results={}'.format(results.total_rows))
    data = dict(results=results)
    return flask.render_template('classes.html', data=data)

@app.route('/relations')
def relations():
    results = BQ_CLIENT.query(
    '''
        SELECT Relation, COUNT(*) As NumImages
        FROM `bdcc-proj.openimages.relations`
        GROUP BY Relation
        ORDER BY Relation ASC
    ''').result()
    logging.info('classes: results={}'.format(results.total_rows))
    data = dict(results=results)
    return flask.render_template('relations.html',data=data)

@app.route('/image_info')
def image_info():
    image_id = flask.request.args.get('image_id')
    # TODO
    
    results = BQ_CLIENT.query(
    '''
     
    SELECT DISTINCT class1.description,
    rel.relation,
    class2.description, 
    ImageID 
    FROM bdcc-proj.openimages.relations rel
    INNER JOIN bdcc-proj.openimages.image_labels USING(ImageId)
    INNER JOIN bdcc-proj.openimages.classes class1 ON class1.label = rel.Label1
    INNER JOIN bdcc-proj.openimages.classes class2 ON class2.label = rel.Label2
    Where ImageID = "{0}" 


    '''.format(image_id)
    ).result()
    results1 =  BQ_CLIENT.query(
    '''
    SELECT Description
    from bdcc-proj.openimages.classes   
    INNER JOIN bdcc-proj.openimages.image_labels USING(Label)
    Where ImageID = "{0}"
    ORDER BY Description ASC
    '''.format(image_id)
    )
    #logging.info('image_info: image_id={}, results={}'\
    #       .format(image_id, results.total_rows))
    data = dict(image_id=image_id,
                results=results, results1 = results1) 
    return flask.render_template('image_id.html',data = data)

@app.route('/image_search')
def image_search():
    description = flask.request.args.get('description')
    image_limit = flask.request.args.get('image_limit', default=10, type=int)
    results = BQ_CLIENT.query(
    '''
        SELECT ImageId
        FROM `bdcc-proj.openimages.image_labels`
        JOIN `bdcc-proj.openimages.classes` USING(Label)
        WHERE Description = '{0}' 
        ORDER BY ImageId
        LIMIT {1}  
    '''.format(description, image_limit)
    ).result()
    logging.info('image_search: description={} limit={}, results={}'\
           .format(description, image_limit, results.total_rows))
    data = dict(description=description, 
                image_limit=image_limit,
                results=results)
    return flask.render_template('image_search.html', data=data)

@app.route('/relation_search')
def relation_search():
    class1 = flask.request.args.get('class1', default='%')
    relation = flask.request.args.get('relation', default='%')
    class2 = flask.request.args.get('class2', default='%')
    image_limit = flask.request.args.get('image_limit', default=10, type=int)
    results = BQ_CLIENT.query(
    '''
        SELECT DISTINCT rel.ImageID, 
        class1.description,
        rel.relation,
        class2.description
        From `bdcc-proj.openimages.relations` rel
        INNER JOIN `bdcc-proj.openimages.classes` class1 ON class1.Label = rel.Label1 
        INNER JOIN `bdcc-proj.openimages.classes` class2 ON class2.Label = rel.Label2
        WHERE class1.description LIKE "{0}" AND 
        rel.relation LIKE "{1}" AND
        class2.description LIKE "{2}"
        ORDER BY ImageID ASC
        LIMIT {3}
    '''.format(class1,relation,class2,image_limit)
    ).result()
    logging.info('relation_search: class1={} relation={} class2={} limit={}, results={}'\
           .format(class1, relation, class2, image_limit, results.total_rows))
    data = dict(class1=class1,
                relation = relation,
                class2 = class2, 
                image_limit=image_limit,
                results=results)
    return flask.render_template('relation_search.html', data = data)

@app.route('/image_search_multiple')
def image_search_multiple():
    descriptions = flask.request.args.get('descriptions').split(',')
    image_limit = flask.request.args.get('image_limit', default=10, type=int)
    # TODO
    results = BQ_CLIENT.query(
    '''
        SELECT ImageId, ARRAY_AGG(Description), COUNT(Description) as descr 
        FROM `bdcc-proj.openimages.image_labels`
        JOIN `bdcc-proj.openimages.classes` USING(Label)
        WHERE Description IN UNNEST({0}) 
        GROUP BY ImageId
        ORDER BY descr desc, ImageId 
        LIMIT {1}
    '''.format(descriptions, image_limit)
    ).result()
    logging.info('image_search_multiple: description={} limit={}, results={}'\
           .format(descriptions, image_limit, results.total_rows))
    data = dict(description=descriptions, 
                image_limit=image_limit,
                results=results)
    return flask.render_template('image_search_multiple.html', data = data)

@app.route('/image_classify_classes')
def image_classify_classes():
    with open(app.root_path + "/static/tflite/my_dict.txt", 'r') as f:
        data = dict(results=sorted(list(f)))
        return flask.render_template('image_classify_classes.html', data=data)

def detect_labels(uri):
    """Detects labels in the file located in Google Cloud Storage or on the
    Web."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri

    response = client.label_detection(image=image)
    labels = response.label_annotations

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
                
    return labels
 
@app.route('/image_classify', methods=['POST'])
def image_classify():
    files = flask.request.files.getlist('files')
    min_confidence = flask.request.form.get('min_confidence', default=0.25, type=float)
    results = []
    if len(files) > 1 or files[0].filename != '':
        for file in files:
            blob = storage.Blob(file.filename, APP_BUCKET)
            blob.upload_from_file(file, blob, content_type=file.mimetype)
            blob.make_public()
            classifications = detect_labels("https://storage.googleapis.com/bdcc-proj.appspot.com/"+file.filename)
            logging.info('image_classify: filename={} blob={} classifications={}'\
                .format(file.filename,blob.name,classifications))
            results.append(dict(bucket=APP_BUCKET,
                                filename=file.filename,
                                classifications=classifications))

    data = dict(bucket_name=APP_BUCKET.name, 
                min_confidence=min_confidence, 
                results=results)
    return flask.render_template('image_classify.html', data=data)

'''
def image_classify():
    files = flask.request.files.getlist('files')
    min_confidence = flask.request.form.get('min_confidence', default=0.25, type=float)
    results = []
    if len(files) > 1 or files[0].filename != '':
        for file in files:
            classifications = TF_CLASSIFIER.classify(file, min_confidence)
            blob = storage.Blob(file.filename, APP_BUCKET)
            blob.upload_from_file(file, blob, content_type=file.mimetype)
            blob.make_public()
            logging.info('image_classify: filename={} blob={} classifications={}'\
                .format(file.filename,blob.name,classifications))
            results.append(dict(bucket=APP_BUCKET,
                                filename=file.filename,
                                classifications=classifications))
    
    data = dict(bucket_name=APP_BUCKET.name, 
                min_confidence=min_confidence, 
                results=results)
    return flask.render_template('image_classify.html', data=data)
'''


if __name__ == '__main__':
    # When invoked as a program.
    logging.info('Starting app')
    app.run(host='0.0.0.0', port=8080, debug=True)