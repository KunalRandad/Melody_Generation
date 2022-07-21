from tensorflow import keras as keras
from preprocess import generating_training_sequences, SEQUENCELENGTH

OUTPUT_UNITS = 32
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
NUM_UNITS = [256]
EPOCHS = 5
BATCH_SIZE = 128
SAVE_MODEL_PATH = r'C:\Users\91702\Desktop\ML_projects\Melody_generation\essen\europa\deutschl\Generated_melodies\\allerkbd\\model_allerkbd.h5'
def build_model(output_units, num_units, loss, learning_rate):
    # create the architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(output_units, activation = "softmax")(x)
    model = keras.Model(input,output)

    # compile model
    model.compile(loss = loss,
                  optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics = ["accuracy"])
    model.summary()
    return model

def train():

    # generate the training sequence
    inputs, targets = generating_training_sequences(SEQUENCELENGTH)
    print(inputs)
    print(inputs.shape)

    # build the network
    model = build_model(output_units = OUTPUT_UNITS, num_units = NUM_UNITS, loss = LOSS, learning_rate = LEARNING_RATE)

    # train the model
    model.fit(inputs, targets, epochs = EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(SAVE_MODEL_PATH)

if __name__=="__main__":
    train()