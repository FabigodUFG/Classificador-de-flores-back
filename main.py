from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import os
from flask_cors import CORS

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configuração do Flask e CORS
app = Flask(__name__)
CORS(app, origins="*")

# Carregar o modelo TFLite
MODEL_PATH = "model/my_flower_model.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

# Obter os detalhes de entrada e saída do modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Classes de flores
CLASS_NAMES = ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips']

# Função para realizar a inferência
def predict(input_image):
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # Criar pasta temporária e salvar a imagem
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)

    try:
        # Preprocessar a imagem
        img = tf.keras.utils.load_img(file_path, target_size=(180, 180))
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)

        # Realizar a predição
        predictions = predict(img_array)
        score = tf.nn.softmax(predictions)

        # Determinar a classe com maior probabilidade
        predicted_class = CLASS_NAMES[np.argmax(score.numpy())]
        confidence = np.max(score.numpy()) * 100

        # Retornar o resultado
        return jsonify({
            "class": predicted_class,
            "confidence": round(float(confidence), 2),
            "predictions": score.numpy().tolist()
        })

    finally:
        # Remover arquivo temporário
        if os.path.exists(file_path):
            os.remove(file_path)

# Iniciar a aplicação Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
