import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, Dense, Dropout, BatchNormalization

def positional_encoding(position, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / tf.math.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    angle_rads = get_angles(tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                            d_model)
    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)

def create_look_ahead_mask(size, size2=None):
    if size2 is None:
        size2 = size
    mask = 1 - tf.linalg.band_part(tf.ones((size, size2)), -1, 0)
    return mask

class GalformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_k, dense_dim, dropout_rate=0.1):
        super(GalformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(
            num_heads=num_heads, key_dim=d_k, dropout=dropout_rate,
            kernel_initializer='he_normal'
        )
        self.ffn = tf.keras.Sequential([
            Dense(dense_dim, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization(),
            Dense(d_model, kernel_initializer='he_normal'),
            BatchNormalization()
        ])
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.dropout_ffn = Dropout(dropout_rate)

    def call(self, x, training):
        attn_output = self.mha(query=x, value=x)
        out1 = self.bn1(tf.add(x, attn_output), training=training)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout_ffn(ffn_output, training=training)
        return self.bn2(tf.add(ffn_output, out1), training=training)

class GalformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_k, dense_dim, seq_len, dropout_rate=0.1):
        super(GalformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.lin_input = Dense(d_model, activation=\"relu\")
        self.pos_encoding = positional_encoding(seq_len, d_model)
        self.enc_layers = [
            GalformerEncoderLayer(d_model, num_heads, d_k, dense_dim, dropout_rate)
            for _ in range(num_layers)
        ]

    def call(self, x, training):
        x = self.lin_input(x)
        seq_len_actual = tf.shape(x)[1]
        x += self.pos_encoding[:, :seq_len_actual, :]
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training)
        return x

def build_galformer_model(seq_len, num_features, d_model=256, num_heads=8, num_layers=4, dense_dim=1024):
    \"\"\"
    Builds an optimized Keras model using the Galformer Encoder logic for Time-Series prediction.
    \"\"\"
    inputs = tf.keras.Input(shape=(seq_len, num_features))
    
    d_k = d_model // num_heads
    encoder = GalformerEncoder(num_layers, d_model, num_heads, d_k, dense_dim, seq_len)
    x = encoder(inputs)
    
    # We take the final state or flatten
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Predict continuous value (e.g. next price step / risk score)
    outputs = Dense(1, activation='linear')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=\"GalformerFlashCrashPredictor\")

if __name__ == '__main__':
    model = build_galformer_model(seq_len=60, num_features=5)
    model.summary()
