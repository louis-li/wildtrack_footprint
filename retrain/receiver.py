import numpy as np
import cv2 
import time, datetime
import paho.mqtt.client as mqtt
import os

# Connect to queue
LOCAL_MQTT_HOST="mosquitto"
LOCAL_MQTT_PORT=1883
TOPIC_WILDTRACK_ALL ="footprint/+"
TOPIC_WILDTRACK_POSITIVE ="footprint/positive"
TOPIC_WILDTRACK_NEGATIVE ="footprint/negative"
TOPIC_WILDTRACK_NOLABEL ="footprint/nolabel"

labels = ['African elephant', 'African lion', 'Amur Tiger', 'Bengal Tiger', 
          'Black Rhino', 'Bongo', 'Cheetah', 'Jaguar', 'Leopard', 'Lowland Tapir',
          'Otter', 'Puma', 'White Rhino']


def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        client.subscribe(TOPIC_WILDTRACK_ALL)

def on_message(client,userdata, msg):
    global labels
    try:
        print(f"Message received!")
        data_root_dir = 'data'
        filename = '{}.jpg'.format(str(time.time()))

        nparr = np.frombuffer(msg.payload, np.uint8)

        if msg.topic == TOPIC_WILDTRACK_NOLABEL:
            print(f"Message received on topic {msg.topic}!")
            image_dir = f"{data_root_dir}/nolabel"
            image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        else:
            print(f"Message received on topic {msg.topic}!")
            label = labels[int(nparr[0])]
            today = datetime.date.today().strftime("%Y%m%d")
            data_dir = f"{data_root_dir}/{today}/{label}"
            
            os.makedirs(data_dir, exist_ok=True) 

            image = cv2.imdecode(nparr[1:], cv2.IMREAD_UNCHANGED)
            
        save_file = os.path.join(data_dir,filename)
        ret = cv2.imwrite(save_file, image)
        print(f"Image saved: {save_file}")
    except:
        print("Unexpected error")


local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, port=LOCAL_MQTT_PORT, keepalive=60)
local_mqttclient.on_message = on_message

# go into a loop
local_mqttclient.loop_forever()

