# music-generator-using-rnn
This is a machine learning project. The goal is to create a recurrent neural network model that can be used to generate new music. The default model is trained on 1800 different musics, producing a decent model for generating musics.

### Installation
To run this script you will need:
Python version >= 3.7, Tensorflow 2.x and music21

### How to use
You can simply generate a music using a default model by running
```
python music_generator.py
```
the program will ask you to type in starting notes and name for the generated music. Then it will output a midi file.

### Train
to train the model further you would need to follow the following example
```
# create train.py
from music_generator import parse_midi, GenerateMusic
import os

musics = []
dataset_dir = 'your_dataset_directory'

for file in os.listdir(dataset_dir):
    # parse each midi file into notes and chords
    musics.append(parse_midi(os.path.join(dataset_dir, file)))

# train the model
weights = GenerateMusic().train([note for music in musics for note in music], epochs=50)

# build new model from trained model with batch size of 1
model = GenerateMusic().build_model(1)
model.set_weights(weights)

# generate music
GenerateMusic().generate_music(['C4', 'D4'], 'name', model)
```
