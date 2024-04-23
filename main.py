import json
import rerun as rr

from confluent_kafka import Consumer, KafkaError, KafkaException

from minio import Minio
from PIL import Image
import numpy as np
from io import BytesIO

minio_cli = Minio(
        "192.168.0.110:8000",
        access_key="root",
        secret_key="root123123",
        secure=False 
)


# Kafka consumer configuration
conf = {
    'bootstrap.servers': '192.168.0.171:31092',   
    'group.id': 'predictions',                    
    'auto.offset.reset': 'earliest'               
}

# Create Kafka consumer
consumer = Consumer(conf)

# Subscribe to the 'predictions' topic
consumer.subscribe(['predictions'])

def objects_to_rerun(json_data) -> rr.Boxes2D:
    boxes = []
    labels = []
    class_ids = []

    for obj in json_data["objects"]:
        boxes.append(obj["box"])
        labels.append(f"{obj['name']}_{obj['tracker_id']}")
        class_ids.append(obj["class"])

    return rr.Boxes2D(
        array=boxes,
        labels=labels,
        array_format=rr.Box2DFormat.XYXY,
        class_ids=class_ids,
    )


def zones_to_rerun(json_data) -> rr.LineStrips2D:
    polygons = []
    polygons_ids = []

    for zone in json_data["zones"]:
        polygons_ids.append(zone["id"])
        polygons.append(zone["polygon"])

    return rr.LineStrips2D(polygons, labels=polygons_ids)


def lines_to_rerun(json_data) -> rr.LineStrips2D:
    lines = []
    line_ids = []
    for line in json_data["lines"]:
        lines.append([line["start"], line["end"]])
        line_ids.append(f'{line["id"]} (In: {line["in_count"]}, Out: {line["out_count"]})')

    return rr.LineStrips2D(lines, labels=line_ids)


def load_image_from_minio(minio_client, bucket_name, object_name):
    image_object = minio_client.get_object(bucket_name, object_name)
    image_data = image_object.read()
    image_pil = Image.open(BytesIO(image_data))
    image_np = np.array(image_pil)
    
    return image_np


def images_to_rerun(json_data):
    bucket_name = "frames"
    object_name = json_data["video_frame"]["id"]
    image = load_image_from_minio(minio_cli, bucket_name, object_name)

    return rr.Image(image).compress(jpeg_quality=30)

# def queue_time_to_rerun(json_data):
#     queue_times = []

#     for queue in json_data["queue_time"]:
#         queue_times.append(queue)

def licence_plates_to_rerun(json_data):
    boxes = []
    labels = []

    if json_data["licence_plates"] is not None:
        for obj in json_data["licence_plates"]:
            boxes.append(obj["region"]["xyxy"])
            labels.append(f"{obj['text']}")
    else:
        return
    
    return rr.Boxes2D(
        array=boxes,
        labels=labels,
        array_format=rr.Box2DFormat.XYXY,
    )


def crowd_to_rerun(json_data):
    points = []

    for coordinate in json_data["points"]:
        points.append(coordinate)

    return rr.Points2D(
        positions=points,
    )

rr.init("rerun_example_box2d", spawn=True)


MODE = 'LPR' # OBJ, LPR, CC

try:
    while True:
        # Poll for new messages
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                # End of partition
                print('%% %s [%d] reached end at offset %d\n' %
                      (msg.topic(), msg.partition(), msg.offset()))
            elif msg.error():
                raise KafkaException(msg.error())
        else:
            # Process the received message
            pred = json.loads(msg.value().decode('utf-8'))

            if MODE == 'OBJ':
                objs = objects_to_rerun(pred)
                zns = zones_to_rerun(pred)
                lns = lines_to_rerun(pred)
                img = images_to_rerun(pred)

                rr.log("object_detection/zones", zns)
                rr.log("object_detection/lines", lns)
                rr.log("object_detection/images", img)
                rr.log("object_detection/bbox", objs)

            elif MODE == 'LPR':
                lcp = licence_plates_to_rerun(pred)
                img = images_to_rerun(pred)

                rr.log("licence_plate_recognition/images", img)
                rr.log("licence_plate_recognition/bbox", lcp)

            elif MODE == 'CC':
                img = images_to_rerun(pred)
                ccp = crowd_to_rerun(pred)

                rr.log("crowd_counting/images", img)
                rr.log("crowd_counting/points", ccp)



except KeyboardInterrupt:
    pass

finally:
    # Clean up the consumer
    consumer.close()
