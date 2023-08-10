from flask import Flask, request, jsonify
import tensorflow as tf
from utils import *


app = Flask(__name__)
def load_model():
    model=tf.keras.models.load_model('LSTM_model.h5')
    return model

def generate_text(model, start_string, generation_length=1000):
    input_eval = [char_to_index[i] for i in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()

    for i in range(generation_length):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(index_to_char[predicted_id])

    return start_string + ''.join(text_generated)


@app.route('/api/generatetext', methods=['GET'])
def handle_generate_text():
    d = {}
    start_string = str(request.args.get('start_string'))
    model = load_model()  # Function to load the model
    output = generate_text(model, start_string)
    d['output'] = output
    return jsonify(d)


if __name__ == "__main__":
    app.run()

