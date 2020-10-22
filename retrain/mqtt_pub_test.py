import cv2 
import paho.mqtt.client as mqtt
import time

def init_mqtt(host, port):
    mqttclient = mqtt.Client()
    mqttclient.connect(host, port=port, keepalive=60)
    return mqttclient
    
def publish_image(mqttclient, img, class_name, topic ="footprint/positive"):
    from datetime import datetime
    labels = {'African elephant':0,
          'African lion':1,
          'Amur Tiger':2,
          'Bengal Tiger':3, 
          'Black Rhino':4,
          'Bongo':5,
          'Cheetah':6,
          'Jaguar':7,
          'Leopard':8,
          'Lowland Tapir':9, 
          'Otter':10,
          'Puma':11,
          'White Rhino':12} 
    img_string = cv2.imencode('.jpg', img)[1].tostring()
    label_id = labels[class_name]

    # merge payload
    payload = b"".join([chr(label_id).encode('utf-8'),img_string])
    
    mqttclient.publish(topic,payload=payload)
    print(f"Image published at {str(datetime.now())}!")

# sample image
img = cv2.imread('data/train/Bongo/M Kalama/IMG_5945.JPG',cv2.IMREAD_COLOR)
class_name = "Bongo"

mqttclient = init_mqtt("ubuntu-gpu2.eastus.cloudapp.azure.com", 1883)

while True:
    publish_image(mqttclient, img, class_name)
    time.sleep(60)

