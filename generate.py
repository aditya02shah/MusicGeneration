import tensorflow as tf
import tqdm
import mitdeeplearning as mdl
from IPython import display as ipythondisplay
import soundfile as sf

songs = mdl.lab1.load_training_data()
songs_joined = "\n\n".join(songs)

# Finding all unique characters in the joined string
vocab = sorted(set(songs_joined))
char_to_index = dict((j,i) for i,j in enumerate(vocab))

index_to_char = dict((i,j) for i,j in enumerate(vocab))


def generate_text(model, start_string, generation_length=1000):
    input_eval = [char_to_index[i] for i in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()

    for i in (range(generation_length)):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
#         print(predicted_id)
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(index_to_char[predicted_id])
    return (start_string + ''.join(text_generated))
     

model=tf.keras.models.load_model('LSTM_model.h5')
demo_text1 = generate_text(model, start_string="Z", generation_length=1000)

generated_songs = mdl.lab1.extract_song_snippet(demo_text1)
save_path="results"
m = 0 
for i, song in enumerate(generated_songs):
    waveform = mdl.lab1.play_song(song)
    if waveform:
        # Save the waveform to a file
        save_path = f"generated_song_{i}.wav"
        mdl.lab1.save_wav(waveform, save_path)
        print("Generated song", i, "saved as", save_path)
        m += 1
