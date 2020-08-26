from flask import Flask, render_template,request
import flask
import pickle
import numpy as np
import keras
from keras.models import load_model
import tensorflow as tf
import werkzeug
from flask import jsonify
app = Flask(__name__)

label_names = ["Bã rác","Rác hữu cơ","Thực vật","Túi lọc trà","Rác thực vật","Rác giấy","Chai kim loại","Chai thủy tinh","Chai nhựa","Giấy viết","Hộp giấy","Vật dụng bằng nhựa","Lon kim loại","Thùng carton","Bao bì sản phẩm","Bao nhựa","Rác nilon","Đồ gốm","Hộp xốp","Giấy ăn","Khẩu trang y tế","Ly giấy","Bóng đèn cũ","Pin"]

labels = {'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8, '17': 9, '18': 10, '19': 11, '2': 12, '20': 13, '21': 14, '22': 15, '23': 16, '3': 17, '4': 18, '5': 19, '6': 20, '7': 21, '8': 22, '9': 23}

from keras import Model 
import keras
ResNet_model = keras.applications.ResNet152V2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# The last 15 layers fine tune
for layer in ResNet_model.layers[:-15]:
    layer.trainable = False

x = ResNet_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
#x = Flatten()(x)
x = keras.layers.Dense(units=1024, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(units=512, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
output  = keras.layers.Dense(units=len(label_names), activation='softmax')(x)
model = keras.Model(ResNet_model.input, output)
model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])

global graph
graph = tf.get_default_graph() 

model.load_weights('model_weight.h5')



from keras.preprocessing import image
def load_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    return img_tensor

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print(filename)
    new_image = load_image(filename)
    with graph.as_default():
        predictions =  model.predict(new_image)
    index = np.argmax(predictions[0])
    index_label = int(list(labels.keys())[list(labels.values()).index(index)])
    output = "{label:"+str(label_names[index_label]) + "}"
    return jsonify(output)

app.run(debug=True)