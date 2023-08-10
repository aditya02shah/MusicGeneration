import mitdeeplearning as mdl
import tensorflow as tf
import numpy as np
from tqdm import tqdm

songs = mdl.lab1.load_training_data()

songs_joined = "\n\n".join(songs) 

# Find all unique characters in the joined string
vocab = sorted(set(songs_joined))

char_to_index = dict((j,i) for i,j in enumerate(vocab))

index_to_char = dict((i,j) for i,j in enumerate(vocab))

def vectorize_string(string):
    vector = []
    for char in string:
#         print(type(char))
        vector .append(char_to_index[char])
    return np.array(vector)

vectorized_songs = vectorize_string(songs_joined)
# print(vectorized_songs.shape)

def generate_text(model, start_string, generation_length=1000):
    input_eval = [char_to_index[i] for i in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()
    tqdm._instances.clear()
    
    for i in tqdm(range(generation_length)):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
#         print(predicted_id)
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(index_to_char[predicted_id])
    return (start_string + ''.join(text_generated))

