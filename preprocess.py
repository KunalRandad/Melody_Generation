import os
import music21 as m21
import json
import tensorflow.keras as keras
import numpy as np
KERN_DATASET_PATH=r'C:\Users\91702\Desktop\ML_projects\Melody_generation\essen\europa\deutschl\\allerkbd\\'
ACCEPTABLE_DURATIONS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]
SAVE_DIR = r'C:\Users\91702\Desktop\ML_projects\Melody_generation\essen\europa\deutschl\Generated_melodies\\allerkbd\\dataset\\'
SINGLE_FILE_DATASET = r'C:\Users\91702\Desktop\ML_projects\Melody_generation\essen\europa\deutschl\Generated_melodies\\allerkbd\\file_dataset'
SEQUENCELENGTH = 64
MAPPING_PATH = r'C:\Users\91702\Desktop\ML_projects\Melody_generation\essen\europa\deutschl\Generated_melodies\\allerkbd\\mapping.json'
def has_acceptable_duration(song,acceptable_duration):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_duration:
            return False
    return True

def encode_song(song, time_step=0.25):
    encoded_song=[]
    for event in song.flat.notesAndRests:
        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert the note/rest into timeseries notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:  # the first symbol would be a charahter followed by "_" for rest of the repetations
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to a string
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song
def load_songs_in_kern(datapath):
    #load data from all the folders in datapath with music21
    songs=[]
    for path,subdir,files in os.walk(datapath):

        for file in files:
            #print(file)
            if file[-3:]=="krn":
                #print("hi")
                song = m21.converter.parse(os.path.join(path,file))
                songs.append(song)
    return songs

def transpose(song):
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
    # get interval for transposition eg Bmaj->Cmaj
    if key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
    # transpose song by calculated interval
    transposed_song = song.transpose(interval)
    return transposed_song
def preprocess(datapath):
    #load the songs
    print("Loading songs")
    songs = load_songs_in_kern(datapath)
    print(f"Loaded {len(songs)} songs.")
    for i, song in enumerate(songs):
    # filter out songs of unacceptable duration
      if not has_acceptable_duration(song, ACCEPTABLE_DURATIONS):
          continue
    #transpose song to Cmaj/Amin
      song = transpose(song)

    #encode songs as music timeseries representation
      encoded_song = encode_song(song)

    #save songs to text file
      save_path = os.path.join(SAVE_DIR, str(i))
      with open(save_path, "w") as fp:
        fp.write(encoded_song)

def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
        return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ "*sequence_length
    songs =""

    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path,file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter
    songs = songs[:-1]  # Because we dont want space after last "/"
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs

    # LOAD ENCODED SONGS AND ADD DELIVERIES


def create_mapping(songs, mapping_path):
    mappings = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # create mappings
    for i,symbol in enumerate(vocabulary):
        mappings[symbol] = i
    # save vocabulary in a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent = 4)

def convert_songs_to_int(songs):
    int_songs =[]
    # load the created mapping
    with open(MAPPING_PATH,"r") as fp:
        mappings = json.load(fp)

    # cast song string to a list
    songs = songs.split()
    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs

def generating_training_sequences(sequence_length):

    # load and map songs to int
    songs = load(SINGLE_FILE_DATASET)
    # print("here1")
    int_songs = convert_songs_to_int(songs)
    # print("here2")
    inputs = []
    targets = []

    # generate the training sequences
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
        # print("here i")
    # one-hot encode the sequences
    vocabulary_size = len(set(int_songs))
    # print("here 4")
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    # print("here 5")
    targets = np.array(targets)

    return inputs,targets


def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCELENGTH)
    songs = create_mapping(songs, MAPPING_PATH)
    inputs, targets = generating_training_sequences(SEQUENCELENGTH)
    # a=1

if __name__ == "__main__":
    main()