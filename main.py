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
import warnings
warnings.filterwarnings("ignore")



MAX_SEQ_LENGTH = 30
NUM_FEATURES = 768
IMG_SIZE = 800
dense_dim = 1024
num_heads = 12


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


class RealTimePredictor:
    def __init__(self, model):
        self.model = model
        self.mobilenet = MobileNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, pooling='avg')
        self.data_queue = Queue()
        self.predicting = False

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame_processed = preprocess_input(frame_resized)
        features = self.mobilenet.predict(np.expand_dims(frame_processed, axis=0), verbose=0)
        features = layers.Dense(NUM_FEATURES, activation='linear', trainable=False)(features)
        # print('/*/*/*/*', features.shape)
        # print('/*/*/*/*', features[0].shape)
        return features[0]

    def add_to_queue(self, features):
        self.data_queue.put(features)
        if not self.predicting and self.data_queue.qsize() >= MAX_SEQ_LENGTH:
            self.predict()

    def predict(self):
        if self.data_queue.qsize() >= MAX_SEQ_LENGTH:
            self.predicting = True
            # 큐에서 30개의 데이터를 복사
            data_to_predict = list(self.data_queue.queue)[:MAX_SEQ_LENGTH]
            threading.Thread(target=self.run_prediction, args=(data_to_predict,)).start()

    def run_prediction(self, data):
        prediction = self.model.predict(np.expand_dims(data, axis=0), verbose=0)
        print("Prediction:", prediction)
        self.predicting = False
        # 첫 번째 데이터 제거
        if not self.data_queue.empty():
            self.data_queue.get()

    def run(self, video_source):
        cap = cv2.VideoCapture(video_source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            features = self.process_frame(frame)
            self.add_to_queue(features)

        cap.release()

# 사용 예시
model = create_model()
model.load_weights('lstm_transformer_with_model.h5')
model.summary()
predictor = RealTimePredictor(model)

# 비디오 파일 사용 예시
predictor.run('final_video.mp4')
# predictor.run(0)  # 웹캠을 사용하는 경우, 비디오 파일 경로를 넣을 수도 있습니다.