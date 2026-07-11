import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

# --- 1. FUNZIONE DI ERRORE (LOSS) CUSTOM ---
# Abbiamo detto che la città è per il 95% asciutta. Serve la Dice Loss!
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = 1 - dice_coef(y_true, y_pred)
    return bce + dice

# --- 2. ARCHITETTURA U-NET ---
def build_unet(input_shape=(256, 256, 11)):
    inputs = layers.Input(shape=input_shape)
    
    # --- ENCODER (La Discesa) ---
    # Livello 1
    c1 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(0.1)(p1) # Previene l'overfitting
    
    # Livello 2
    c2 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    c2 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(0.1)(p2)
    
    # --- BOTTLENECK (Il Cervello Centrale) ---
    c3 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    c3 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    
    # --- DECODER (La Salita) ---
    # Livello Risalita 1
    u4 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    concat4 = layers.Concatenate()([u4, c2]) # SKIP CONNECTION
    c4 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(concat4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)
    c4 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)
    
    # Livello Risalita 2
    u5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    concat5 = layers.Concatenate()([u5, c1]) # SKIP CONNECTION
    c5 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(concat5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)
    c5 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)
    
    # --- OUTPUT ---
    # Attivazione Sigmoide per avere probabilità da 0.0 a 1.0
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# --- 3. INIZIALIZZAZIONE E COMPILAZIONE ---
# Creiamo il modello specificando i nostri 11 canali in ingresso
modello_allagamento = build_unet(input_shape=(256, 256, 11))

# Compiliamo usando la nostra loss avanzata e tenendo d'occhio il coefficiente di Dice
modello_allagamento.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=bce_dice_loss,
    metrics=[dice_coef, tf.keras.metrics.BinaryAccuracy()]
)

# Stampa un riassunto bellissimo di tutta l'architettura
modello_allagamento.summary()