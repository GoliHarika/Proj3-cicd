import base64
import boto3
import json
import time
import logging

# AWS Clients
s3_client = boto3.client('s3')
rekognition_client = boto3.client('rekognition')
textract_client = boto3.client('textract')
dynamodb = boto3.resource('dynamodb')

# DynamoDB Table
table = dynamodb.Table('proj3-participation-sdk')

# S3 Bucket Name
bucket_name = 'proj3-group12-bucket-sdk'

# Set up logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def decode_and_upload_image(event_body):
    """Decodes base64 image, saves it to S3, and returns the S3 key."""
    image_base64 = event_body.get('image')
    if not image_base64:
        logger.error("No image data provided in the event.")
        return None

    try:
        image_data = base64.b64decode(image_base64)
        name = event_body.get('name', 'default_name')
        timestamp = int(time.time())
        file_name = f"{name}_{timestamp}.jpg"
        date_folder = event_body.get('date', 'unknown_date')
        s3_key = f"proj3/proj3-images/{date_folder}/{file_name}"

        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=image_data, ContentType='image/jpeg')
        logger.info(f"Image uploaded to S3: {s3_key}")
        return s3_key
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        return None

def lambda_handler(event, context):
    cors_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': '*',
        'Access-Control-Allow-Methods': 'POST,OPTIONS'
    }

    # Handle CORS preflight request
    if event.get("httpMethod", "") == "OPTIONS":
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps({'message': 'CORS preflight handled'})
        }

    try:
        event_body = json.loads(event['body']) if 'body' in event else event
        logger.info(f"Parsed event body: {json.dumps(event_body, indent=2)}")

        key_upload = decode_and_upload_image(event_body) if 'image' in event_body else None
        key_face, key_name = [], []

        def retrive_image(s3_key):
            try:
                response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
                image_bytes = response['Body'].read()
                return base64.b64encode(image_bytes).decode('utf-8')
            except Exception as e:
                logger.error(f"Error fetching image {s3_key}: {str(e)}")
                return None

        def extract_text_from_image(name_key):
            try:
                image_base64 = retrive_image(name_key)
                if not image_base64:
                    return []
                image_bytes = base64.b64decode(image_base64)
                textract_response = textract_client.detect_document_text(Document={'Bytes': image_bytes})
                return [block['Text'] for block in textract_response.get('Blocks', []) if block['BlockType'] == 'LINE']
            except Exception as e:
                logger.error(f"Error processing name image {name_key}: {str(e)}")
                return []

        def compare_faces(source_image_base64, target_image_base64):
            try:
                source_image = base64.b64decode(source_image_base64)
                target_image = base64.b64decode(target_image_base64)
                response = rekognition_client.compare_faces(
                    SourceImage={'Bytes': source_image},
                    TargetImage={'Bytes': target_image},
                    SimilarityThreshold=85
                )
                face_matches = response.get('FaceMatches', [])
                return bool(face_matches), face_matches[0]['Similarity'] if face_matches else 0.0
            except Exception as e:
                logger.error("Error comparing faces: %s", str(e))
                return False, 0.0

        def store_participation(name, date, email, participation):
            try:
                table.put_item(Item={
                    'name': name,
                    'date': date,
                    'email': email,
                    'participation': participation
                })
                logger.info(f"Stored participation record: {name}, {date}, {email}, {participation}")
            except Exception as e:
                logger.error(f"Error storing participation record: {str(e)}")

        try:
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='proj3/proj3-images/')
            for obj in response.get('Contents', []):
                key = obj['Key']
                if "face" in key:
                    key_face.append(key)
                elif "name" in key:
                    key_name.append(key)
        except Exception as e:
            logger.error(f"Error listing objects from S3: {str(e)}")
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({
                    'message': 'Error listing S3 objects',
                    'error': str(e)
                })
            }

        matching_name = False
        extracted_text = []
        if key_name:
            for name_key in key_name:
                extracted_text = extract_text_from_image(name_key)
                if event_body.get('name') and any(event_body['name'].lower() in text.lower() for text in extracted_text):
                    matching_name = True
                    break

        uploaded_image_base64 = retrive_image(key_upload)
        matching_face = False
        face_similarity_scores = []

        for face_key in key_face:
            target_image_base64 = retrive_image(face_key)
            if target_image_base64:
                match, similarity = compare_faces(uploaded_image_base64, target_image_base64)
                if match:
                    matching_face = True
                    face_similarity_scores.append(similarity)

        participation = matching_name or matching_face
        store_participation(
            event_body.get('name', 'unknown'),
            event_body.get('date', 'unknown'),
            event_body.get('email', 'unknown'),
            participation
        )

        return {
            'statusCode': 200 if participation else 400,
            'headers': cors_headers,
            'body': json.dumps({
                'message': 'Match found' if participation else 'No match found',
                'matching_name': matching_name,
                'matching_face': matching_face,
                'similarity_score': face_similarity_scores,
                'extracted_text': extracted_text
            })
        }

    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': cors_headers,
            'body': json.dumps({
                'message': 'Internal server error',
                'error': str(e)
            })
        }