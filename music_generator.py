from music21 import converter, instrument, note, chord, stream
import tensorflow as tf
import json
import numpy as np
import os
keras = tf.keras


BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def parse_midi(file_dir):
    notes = []
    try:
        midi = converter.parse(file_dir)
        parts = instrument.partitionByInstrument(midi)
        notes_to_parse = None

        if parts:  # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:

            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    except Exception as e:
        print(e)
        pass

    return notes


def split_input_target(notes):
    input_notes = notes[:-1]
    output_notes = notes[1:]
    return input_notes, output_notes


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


class GenerateMusic(object):

    def __init__(self):

        # Load training dataset
        self.dataset = json.loads(
            open(os.path.join(BASE_DIR, 'music_data.json'), 'r').read()
        )
        # Load dictionary for changing notes and chords into int
        self.note_dict = json.loads(
            open(os.path.join(BASE_DIR, 'note_dict.json'), 'r').read()
        )
        # Load list of existing notes and chords
        self.note_list = json.loads(
            open(os.path.join(BASE_DIR, 'note_list.json'), 'r').read()
        )

    def notes_to_ints(self, music_notes):

        notes_as_ints = []

        for music_note in music_notes:
            try:
                notes_as_ints.append(self.note_dict[music_note])
            except KeyError:
                notes_as_ints.append(0)

        return np.array(notes_as_ints)

    def int_to_note(self, int_val):

        return self.note_list[int_val]

    def build_model(self, batch_size):

        model = keras.Sequential([
            keras.layers.Embedding(len(self.note_list), 256, batch_size=batch_size),
            keras.layers.LSTM(
                1024,
                return_sequences=True,
                stateful=True,
            ),
            keras.layers.Dense(len(self.note_list))
        ])

        return model

    def train(self, data, epochs=50):

        '''train model with your data

        :param

        data (list): 1D array of notes and chords that were extract from midi_to_list function
        epoch (int): Number of epoch to train on

        :return: weights of a trained model
        '''

        notes_as_ints = self.notes_to_ints(data)

        seq_length = 150
        print(notes_as_ints)
        dataset = tf.data.Dataset.from_tensor_slices(notes_as_ints)
        sequences = dataset.batch(seq_length + 1, drop_remainder=True)
        data = sequences.map(split_input_target)
        data = data.shuffle(10000).batch(64, drop_remainder=True)

        train_model = self.build_model(64)
        train_model.load_weights(
            os.path.join(BASE_DIR, 'pretrained/weights/variables')
        )

        train_model.compile(optimizer='adam', loss=loss)
        train_model.fit(data, epochs=int(epochs))

        return train_model.get_weights()

    def default_model(self):

        model = tf.keras.models.load_model(
            os.path.join(BASE_DIR, 'pretrained/music_generator_model.h5')
        )

        return model

    def generate_music(self, start_note, name, model, complexity=2.0, num_generate=100):
        # Generate music from model

        # convert note in to int
        input_eval = self.notes_to_ints(start_note)
        input_eval = tf.cast(input_eval, dtype=tf.float32)
        # add batch dimention
        input_eval = tf.expand_dims(input_eval, 0)

        # store prediction here
        note_generated = []

        # reset model's memory
        model.reset_states()
        for i in range(num_generate):

            predictions = model(input_eval)

            predictions = tf.squeeze(predictions, 0)

            # predict the note returned by the model
            predictions = predictions / complexity
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            input_eval = tf.expand_dims([predicted_id], 0)

            note_generated.append(self.int_to_note(predicted_id.tolist()))

        offset = 0
        output_notes = []

        # Generate MIDI file from prediction

        for pattern in note_generated:

            if ('.' in pattern) or pattern.isdigit():  # Check if it's a chord
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Guitar()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)

            else:  # Check if it's a note
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Guitar()
                output_notes.append(new_note)
            # increase offset each iteration so that notes do not stack
            offset += 0.5

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp=os.path.join(BASE_DIR, f'{name}.mid'))

    def generate_from_input(self):

        starting_notes = []
        print('Accepted note format is note A-G follow by number of octave, for example, "A4", "C3"')
        inp_note = input('Please type the starting note >')
        starting_notes.append(inp_note)

        while True:
            more_inp = input('Do you wish to add more starting notes [y/N] >')

            if str(more_inp) in ['Y', 'y']:
                inp_note = input('Please type more note >')
                starting_notes.append(inp_note)
            elif str(more_inp) in ['N', 'n']:
                break
            else:
                print('please type y or N')

        name = str(input('Music name >'))
        model = self.default_model()
        self.generate_music(starting_notes, name, model)


if __name__ == '__main__':
    GenerateMusic().generate_from_input()
