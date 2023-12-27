from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_addons as tfa
from focal_loss import BinaryFocalLoss
import tensorflow as tf
import cv2
import numpy as np
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.models import load_model
import threading
from queue import Queue

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
import warnings
warnings.filterwarnings("ignore")
import main


MAX_SEQ_LENGTH = 15 #30
NUM_FEATURES = 1280 #//768
IMG_SIZE = 224 #800
dense_dim = 1024
num_heads = 12

class RealTimePredictor:
    def __init__(self, model):
        self.model = model
        # self.mobilenet = tf.keras.applications.DenseNet121(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, pooling='avg')
        self.mobilenet = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, pooling='avg')
        self.data_queue = Queue()
        self.predicting = False
        self.latest_prediction = None
        self.lock = threading.Lock()
        self.prediction_started = False
    def process_frame(self, frame):
        global NUM_FEATURES
        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame_resized = frame_resized[:, :, [2, 1, 0]]
        frame_resized = frame_resized/255.
        # frame_processed = preprocess_input(frame_resized)
        features = self.mobilenet.predict(np.expand_dims(frame_resized, axis=0), verbose=0)
        # features = layers.Dense(NUM_FEATURES, activation='linear', trainable=False)(features)
        NUM_FEATURES = features.shape[-1]
        return features[0]

    def add_to_queue(self, features):
        self.data_queue.put(features)
        if not self.predicting and self.data_queue.qsize() >= MAX_SEQ_LENGTH:
            self.predict()

    def predict(self):
        with self.lock:
            if self.data_queue.qsize() >= MAX_SEQ_LENGTH:
                self.predicting = True
                data_to_predict = list(self.data_queue.queue)[:MAX_SEQ_LENGTH]
                threading.Thread(target=self.run_prediction, args=(data_to_predict,)).start()
                self.prediction_started = True

    def run_prediction(self, data):
        with self.lock:
            prediction = self.model.predict(np.expand_dims(data, axis=0), verbose=0)
            self.latest_prediction = prediction[0][0]
            print("Prediction:", 1.0 - prediction)
            self.predicting = False
            if not self.data_queue.empty():
                self.data_queue.get()

    def process_and_predict(self, frame):
        features = self.process_frame(frame)
        self.add_to_queue(features)

class Application:
    def __init__(self, root, video_path, predictor):
        self.root = root
        self.video_path = video_path
        self.predictor = predictor
        self.cap = cv2.VideoCapture(video_path)
        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, 30)

        self.video_label = ttk.Label(root)
        self.video_label.pack(padx=10, pady=10)

        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.line, = self.ax.plot([], [], lw=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(padx=10, pady=10)

        self.ani = animation.FuncAnimation(self.fig, self.update_graph, init_func=self.init_graph, interval=20, blit=True)
        self.xdata = []
        self.ydata = []
        self.graph_length = 100
        self.frame_counter = 0
        self.hline_drawn = False


    def update_video(self):
        # print(217839471324987238957,self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        # if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == 0:
        #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, 30)

        ret, frame = self.cap.read()

        self.frame_counter += 1

        if ret:
            if self.frame_counter % 1 == 0:
                self.predictor.process_and_predict(frame)  # 프레임 처리 및 예측 수행

            # if self.predictor.prediction_started and self.frame_counter >= 30:
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #     frame = cv2.resize(frame, (800, 480))
            # else:
            #     # 그렇지 않다면 검은 화면 표시
            #     frame = np.zeros((480, 800, 3), dtype=np.uint8)

            if self.frame_counter >= 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (800, 480))
            else:
                # 그렇지 않다면 검은 화면 표시
                frame = np.zeros((480, 800, 3), dtype=np.uint8)

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        if not ret:  # 동영상 끝에 도달했을 때
            self.reset_video()
        # if not ret:
        #     # 동영상 끝에 도달했을 때
        #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 동영상을 처음으로 되돌림
        #     ret, frame = self.cap.read()  # 첫 번째 프레임을 다시 읽음
        #     self.frame_counter = 0
        #     self.xdata.clear()
        #     self.ydata.clear()
        #     self.predictor.predicting = False
        #     self.predictor.latest_prediction = None
        #     self.hline_drawn = False
        self.video_label.after(40, self.update_video)

    def reset_video(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_counter = 0
        self.xdata.clear()
        self.ydata.clear()
        with self.predictor.lock:
            self.predictor.predicting = False
            self.predictor.latest_prediction = None
            while not self.predictor.data_queue.empty():
                self.predictor.data_queue.get()
        self.hline_drawn = False

    def __del__(self):
        # 리소스 해제
        if self.cap.isOpened():
            self.cap.release()

    def update_graph(self, frame):
        frame_number = self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 1  # 현재 프레임 번호

        # 최초 30프레임 동안은 예측값을 0으로 설정
        if frame_number < MAX_SEQ_LENGTH:
            prediction = 0
        elif self.predictor.latest_prediction is not None:
            prediction = self.predictor.latest_prediction
            prediction = 1-prediction
        else:
            return [self.line,]

        # 데이터 추가
        self.xdata.append(frame_number)
        self.ydata.append(prediction)

        # 최신 50개 데이터만 유지
        if len(self.xdata) > self.graph_length:
            self.xdata.pop(0)
            self.ydata.pop(0)

        # 그래프 데이터 업데이트
        self.line.set_data(self.xdata, self.ydata)
        # self.ax.axhline(y=0.85, color='red', linestyle='--', linewidth=2)
        if not self.hline_drawn:
            self.ax.axhline(y=0.6, color='red', linestyle='--')
            self.hline_drawn = True
        self.ax.set_xlim(self.xdata[0], self.xdata[-1])  # x축 범위 조정
        self.ax.relim()
        self.ax.autoscale_view()
        return [self.line,]

    def init_graph(self):
        self.ax.set_xlim(0, 10)  # 초기 x축 범위 설정
        self.ax.set_ylim(0, 1)  # 초기 y축 범위 설정
        return [self.line, ]

    def run(self):
        self.update_video()
        tk.mainloop()

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value = inputs['query'], inputs['key'], inputs['value']
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        matmul_qk = tf.matmul(query, key, transpose_b=True)
        attention_weights = tf.nn.softmax(matmul_qk, axis=-1)
        # attention_weights = tf.nn.sigmoid(matmul_qk)
        output = tf.matmul(attention_weights, value)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output

class BiLSTMWithMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, units, num_heads, d_model):
        super(BiLSTMWithMultiHeadAttention, self).__init__()
        self.units = units
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.units, return_sequences=True)
        )
        self.multi_head_attention = MultiHeadAttention(num_heads, d_model)
        self.batchnom = layers.BatchNormalization()

    def call(self, inputs):
        x = self.bilstm(inputs)
        x = self.multi_head_attention({
            'query': x,
            'key': x,
            'value': x
        })
        return x

class RelativePositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, embed_dim):
        super(RelativePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.rel_embeddings = self.add_weight("rel_embeddings", shape=[sequence_length, embed_dim])

    def call(self, x):
        rel_positions = self.rel_embeddings
        # 상대적 위치 정보를 입력 텐서의 차원과 일치하게 만듭니다.
        rel_positions = tf.expand_dims(rel_positions, axis=0)  # Shape: (1, sequence_length, embed_dim)
        # 상대적 위치 정보를 각 샘플에 대해 복제합니다.
        rel_positions = tf.repeat(rel_positions, repeats=tf.shape(x)[0], axis=0)
        return x + rel_positions[:, :tf.shape(x)[1], :self.embed_dim]  # Truncate to match the feature dimension

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.dropout_layer = layers.Dropout(0.23)
        self.attention = layers.MultiHeadAttention(
            # num_heads=num_heads, key_dim=embed_dim, dropout=0.3
            num_heads=num_heads, key_dim=embed_dim, dropout=0.23
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation='relu'),  # tf.nn.gelu
             layers.Dropout(0.23),
             layers.Dense(1024, activation='relu'),
             layers.Dropout(0.3),
             layers.Dense(2048, activation='relu'),
             layers.Dropout(0.3),
             layers.Dense(embed_dim, activation='relu'),
             ]
        )
        self.dense_proj2 = keras.Sequential(
            [layers.Dense(3072, activation='relu')]
        )
        self.dense_proj3 = keras.Sequential(
            [layers.Dense(embed_dim, activation='relu')]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        # attention_output = self.attention(inputs, inputs, attention_mask=mask)
        # proj_input = self.layernorm_1(inputs + attention_output)
        # attention_output = self.dropout_layer(proj_input)
        #
        # output = self.layernorm_2(attention_output)
        # output = self.dense_proj2(output)
        # output = self.dropout_layer(output)
        # output = self.dense_proj3(output)
        # output = self.dropout_layer(output)
        # return output

        inputs = self.layernorm_1(inputs)
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        # proj_input = self.layernorm_1(inputs + attention_output)
        proj_input = self.dropout_layer(attention_output)
        proj_output = self.dense_proj(proj_input)
        output = self.layernorm_2(proj_input + proj_output)
        output = self.dense_proj2(output)
        output = self.dropout_layer(output)
        return self.dense_proj3(output)
def create_model():
    inputs = keras.Input(shape=(None, NUM_FEATURES))
    x = BiLSTMWithMultiHeadAttention(1024, 16, NUM_FEATURES)(inputs)
    relative_positional_encoding = RelativePositionalEncoding(MAX_SEQ_LENGTH, x.shape[-1])
    x = relative_positional_encoding(x)
    x = TransformerEncoder(x.shape[-1], dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.42)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    return model
if __name__ == "__main__":
    # 사용 예시
    model = create_model()
    # model.load_weights('ubi_lstm_transformer.h5')
    # model.load_weights('hockey_mobile_lstm_transformer.h5')
    model.load_weights('./224_15_hockey_mobile_lstm_transformer.h5')
    # Tkinter 윈도우 설정
    root = tk.Tk()
    root.title("Real-time Video and Prediction Graph")
    # model.summary()

    predictor = RealTimePredictor(model)
    app = Application(root, './final_video_short2.mp4', predictor)
    # app = Application(root, 'F_161_0_0_0_0.mp4', predictor)

    app.run()