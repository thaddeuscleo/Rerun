import json
import rerun as rr

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
        line_ids.append(line["id"])

    return rr.LineStrips2D(lines, labels=line_ids)



rr.init("rerun_example_box2d", spawn=True)


from confluent_kafka import Consumer, KafkaError, KafkaException

# Kafka consumer configuration
conf = {
    'bootstrap.servers': '192.168.0.171:31092',     # Kafka broker address
    'group.id': 'predictions',                      # Consumer group ID
    'auto.offset.reset': 'earliest'                 # Start reading at the beginning of the topic
}

# Create Kafka consumer
consumer = Consumer(conf)

# Subscribe to the 'predictions' topic
consumer.subscribe(['predictions'])

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

            objs = objects_to_rerun(pred)
            zns = zones_to_rerun(pred)
            lns = lines_to_rerun(pred)

            rr.log("object_detection/bbox", objs)
            rr.log("object_detection/zones", zns)
            rr.log("object_detection/lines", lns)

except KeyboardInterrupt:
    pass

finally:
    # Clean up the consumer
    consumer.close()
