from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)  

@app.route('/update-path', methods=['POST'])
def update_path():
    try:
        data = request.json
        pdf_path = data.get('pdfPath')

        if not pdf_path:
            return jsonify({'error': 'No PDF path provided'}), 400

        # Pass the PDF path to the Colab model API
        colab_url = "https://colab.research.google.com/github/manrajc13/Byte_chase/blob/main/MCQ%2BCheatSheet%2BoutputPDF.ipynb"  # Replace with your Colab app URL
        colab_response = requests.post(colab_url, json={'pdfPath': pdf_path})

        if colab_response.status_code == 200:
            return jsonify({'message': 'PDF path sent successfully', 'modelResponse': colab_response.json()}), 200
        else:
            return jsonify({'error': 'Failed to get a response from the model', 'details': colab_response.text}), 500

    except Exception as e:
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(port=8080, debug=True)
