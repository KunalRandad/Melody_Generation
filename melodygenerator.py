import tensorflow.keras as keras
import json
import numpy as np
import music21 as m21
from preprocess import SEQUENCELENGTH, MAPPING_PATH

class MelodyGenerator:
    def __init__(self, model_path = r'C:\Users\91702\Desktop\ML_projects\Melody_generation\essen\europa\deutschl\Generated_melodies\\allerkbd\\model_allerkbd.h5'):
      self.model_path = model_path
      self.model = keras.models.load_model(model_path)

      with open(MAPPING_PATH, "r") as fp:
          self._mappings = json.load(fp)
          print(self._mappings)
      self._start_symbols = ["/"]*SEQUENCELENGTH


    def _sample_with_temperature(self, probabilities, temperature):
        # temperature --> infinity means randomly picking one of the values
        # temperature --> 0 means the highest get the probability 1 and other 0
        # temperature --> 1 means probability of choosing certain sample remains same
        print(probabilities.size)

        predictions = np.log(probabilities)/temperature
        probabilities = np.exp(predictions/np.sum(np.exp(predictions)))

        choices = range(len(probabilities))
        # print(sum(probabilities))
        # if (sum(probabilities)<0.9):
        #     print(probabilities)
        probabilities = probabilities*(1/sum(probabilities))
        index = np.random.choice(choices, p=probabilities)
        return index

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        # temperature is a value between 0 to infinity or 1 and we will use this to sample outputs that comes out of the network as a probability distribution
        # seed is a starting sequence for the LSTM to build on

        # creating seed
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        seed =[self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            seed = seed[-max_sequence_length:]   # inorder to take the last part of the seed
            one_hot_seed = keras.utils.to_categorical(seed, num_classes = len(self._mappings))
            one_hot_seed = one_hot_seed[np.newaxis, ...]
            # we do the above step as our model needs input in the format (1 or None or some natural number denoting total number of sequences, sequence_length, num_classes)

            # make a prediction
            probabilities = self.model.predict(one_hot_seed)[0]
            # the first dimension of the model.predict corresponds to the total number of sequences we feed in (1 in our case). Hence, we have [0] at the end in previous line
            output_int = self._sample_with_temperature(probabilities,temperature)

            seed.append(output_int)

            # map output
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]
            # the above line gives a list, so we add a '[0]' at the end to get the value

            if output_symbol == "/":
                break

            melody.append(output_symbol)
        return melody

    def save_melody(self, melody, step_duration = 0.25, format="midi", filename=r'C:\Users\91702\Desktop\ML_projects\Melody_generation\essen\europa\deutschl\Generated_melodies\\allerkbd\\mel.mid'):

        # make a music21 stream
        stream = m21.stream.Stream()
        # parse all the symbols in the melody and create note/rest objects
        start_symbol = None
        step_counter = 1

        for i,symbol in enumerate(melody):
            if symbol !="_" or i+1==len(melody):

                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    else:
                        m21_event = m21.note.Note(int(start_symbol),quarterLength = quarter_length_duration)

                    stream.append(m21_event)

                    step_counter = 1
                start_symbol = symbol
            else:
                step_counter+=1
        # write m21 stream to midi file
        stream.write(format,filename)


if __name__ == "__main__":
    mg = MelodyGenerator()
    seed = "76 _ _ _ 76 _ _ _ 76 _ _ _ 73 _ _ _ 73 _ _ _ 74 _ _ _ _ _ _ _ 71 _ _ _ 71 _ _ _ 73"
    melody = mg.generate_melody(seed, 500, SEQUENCELENGTH, 1)
    print(melody)
    mg.save_melody(melody)



