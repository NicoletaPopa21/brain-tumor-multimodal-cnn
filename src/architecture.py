import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input

class FractionalConv2D(layers.Layer):
    """
    Custom Keras Layer for Fractional Convolution (Caputo/GL inspired).
    Applies a fractional derivative mask to highlight edge anomalies.
    """
    def __init__(self, alpha=0.6, **kwargs):
        super(FractionalConv2D, self).__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        c0, c1, c2 = 1.0, -self.alpha, (self.alpha * (self.alpha - 1.0)) / 2.0
        mask = np.array([[c2, c1, c2], [c1, c0, c1], [c2, c1, c2]], dtype=np.float32)
        channels = input_shape[-1]
        kernel = np.zeros((3, 3, channels, 1), dtype=np.float32)
        for i in range(channels): 
            kernel[:, :, i, 0] = mask
        self.kernel = tf.Variable(initial_value=kernel, trainable=False, name='fractional_kernel')

    def call(self, inputs):
        return tf.nn.depthwise_conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')

    def get_config(self):
        config = super(FractionalConv2D, self).get_config()
        config.update({'alpha': self.alpha})
        return config


def build_fractional_multimodal_model(input_shape=(224, 224, 3)):
    """
    Builds the Dual-Branch Multimodal CNN with Fractional Residual Connections.
    """
    # Pre-trained backbone (Shared Weights)
    base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False)
    base_model.trainable = False

    def build_branch(input_name, branch_suffix):
        inputs = Input(shape=input_shape, name=input_name)
        
        # Fractional Filter & Normalization
        x_frac = FractionalConv2D(alpha=0.6)(inputs)
        x_frac = layers.BatchNormalization(name=f"bn_{branch_suffix}")(x_frac)
        
        # Residual Connection
        x_res = layers.Add(name=f"residual_{branch_suffix}")([inputs, x_frac])
        
        # Feature Extraction
        x = base_model(x_res, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu', name=f"fc_{branch_suffix}")(x)
        return inputs, x

    # Create MRI and CT branches
    in_mri, feat_mri = build_branch("mri_input", "mri")
    in_ct, feat_ct = build_branch("ct_input", "ct")

    # Modality Fusion
    fused = layers.Concatenate(name="feature_fusion")([feat_mri, feat_ct])
    z = layers.Dense(64, activation='relu', name="fusion_reasoning")(fused)
    z = layers.Dropout(0.4)(z)
    
    # Final Decision (Healthy vs Tumor)
    output = layers.Dense(2, activation='softmax', name="final_decision")(z)

    model = models.Model(inputs=[in_mri, in_ct], outputs=output, name="Multimodal_Fractional_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
