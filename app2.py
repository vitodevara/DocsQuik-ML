from flask import Flask, render_template, request, redirect, url_for
from flask_ngrok import run_with_ngrok
from PIL import Image
import os
import pandas as pd
from ultralytics import YOLO

app = Flask(__name__)
run_with_ngrok(app)  # ngrok to create a public URL
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model_dataset = YOLO("best.pt")

def process_image(image_path):
    # Perform inference
    results = model_dataset(image_path)
    
    # Convert results to DataFrame
    boxes_list = results[0].boxes.data.tolist()
    columns = ['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class_id']
    
    for i in boxes_list:
        i[:4] = [round(i, 1) for i in i[:4]]
        i[5] = int(i[5])
        i.append(results[0].names[i[5]])

    columns.append('class_name')
    result_df = pd.DataFrame(boxes_list, columns=columns)
    
    # Filter out 'X' class_name and convert class_name to string
    result_df = result_df[result_df['class_name'] != 'X']
    result_df['class_name'] = result_df['class_name'].astype(str)
    
    return result_df

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            result_df = process_image(filename)
            return render_template('result.html', tables=[result_df], titles=result_df.columns.values)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run()
