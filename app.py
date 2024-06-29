from flask import Flask, render_template, request, send_file
# from GBM.utils.utils import fractal_dimension, entropy, lacunarity
import pickle
import pandas as pd
import os
from PIL import Image
import numpy as np
# import mlflow
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage

app = Flask(__name__)


import cv2
import numpy as np
from skimage import color, io
from skimage.filters import threshold_otsu
from skimage.measure import shannon_entropy
from skimage.util import img_as_ubyte


def label_extraction_func(filename):
    return filename.split('_')[0]




def binarize_image(image):
    if len(image.shape) == 3:
        gray_image = color.rgb2gray(image)
    else:
        gray_image = image
    thresh = threshold_otsu(gray_image)
    binary_image = gray_image < thresh
    return binary_image




def entropy(image):
    return shannon_entropy(image)


def lacunarity(image, window_size):
    def sliding_window(image, window_size):
        for x in range(0, image.shape[0] - window_size + 1, window_size):
            for y in range(0, image.shape[1] - window_size + 1, window_size):
                yield image[x:x + window_size, y:y + window_size]

    lacunarity_values = []
    for window in sliding_window(image, window_size):
        mean = np.mean(window)
        if mean > 0:
            lacunarity_values.append(np.var(window) / mean**2)
    return np.mean(lacunarity_values)


def fractal_dimension(image):
    binary_image = binarize_image(image)
    p = min(binary_image.shape)
    n = 2**np.floor(np.log2(p))

    def boxcount(img, k):
        S = np.add.reduceat(
            np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
            np.arange(0, img.shape[1], k), axis=1)

        return len(np.where((S > 0) & (S < k*k))[0])

    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    counts = [boxcount(binary_image, size) for size in sizes]

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0], sizes, counts



def load_image(image_path):
    image = io.imread(image_path)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return img_as_ubyte(image)


# Set the upload folder
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html', result=None, file_path=None, fd=None, entropy=None, lacunarity=None)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Load the model
            model_path ="model.pkl"
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            
            # Get the file from the form
            img_file = request.files.get('file')
            if not img_file:
                return render_template('index.html', result="No file uploaded", file_path=None, fd=None, entropy=None, lacunarity=None)

            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
            img_file.save(file_path)

            # Read the image
            img = Image.open(file_path)
            img = np.array(img.convert('L'))  # Convert image to grayscale
            
            # Compute features
            fd = fractal_dimension(img)
            en = entropy(img)
            lc = lacunarity(img, window_size=8)
            
            # Ensure features are single numerical values
            if isinstance(fd, tuple):
                fd = fd[0]
            if isinstance(en, tuple):
                en = en[0]
            if isinstance(lc, tuple):
                lc = lc[0]

            # with mlflow.start_run():
            #     mlflow.log_param("fractal_dimension", fd)
            #     mlflow.log_param("entropy", en)
            #     mlflow.log_param("lacunarity", lc)
            #     mlflow.log_metric("fractal_dimension", fd)
            #     mlflow.log_metric("entropy", en)
            #     mlflow.log_metric("lacunarity", lc)
            #     mlflow.sklearn.log_model(model, "model")

            # Create DataFrame and ensure correct types
            test = pd.DataFrame(data={
                'fractal_dimension': [fd],
                'entropy': [en],
                'lacunarity': [lc]
            }).astype(float)
            
            # Predict
            prediction = model.predict(test.values)
            
            # Interpret the result
            result = "GBM-Grade-IV" if prediction[0] == 1 else "Normal"

            return render_template('index.html', result=result, file_path=file_path, fractal_dimension=fd, entropy=en, lacunarity=lc)
        
        except Exception as e:
            return render_template('index.html', result=f"An error occurred: {str(e)}", file_path=None, fractal_dimension=None, entropy=None, lacunarity=None)

    return render_template('index.html', result=None, file_path=None, fd=None, entropy=None, lacunarity=None)

@app.route('/download_report', methods=['GET'])
def download_report():
    result = request.args.get('result')
    fractal_dimension = request.args.get('fractal_dimension')
    entropy = request.args.get('entropy')
    lacunarity = request.args.get('lacunarity')
    file_path = request.args.get('file_path')

    # Create PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Update or add custom styles
    if 'Centered' not in styles:
        styles.add(ParagraphStyle(name='Centered', alignment=1, fontSize=20, spaceAfter=20))
    if 'Heading' not in styles:
        styles.add(ParagraphStyle(name='Heading', fontSize=14, spaceAfter=10, textColor=colors.black))
    if 'BodyText' not in styles:
        styles.add(ParagraphStyle(name='BodyText', fontSize=12, spaceAfter=10))
    else:
        styles['BodyText'].fontSize = 12
        styles['BodyText'].spaceAfter = 10
    if 'Result' not in styles:
        styles.add(ParagraphStyle(name='Result', fontSize=18, spaceAfter=20, textColor=colors.red, fontName="Helvetica-Bold"))
    else:
        styles['Result'].fontSize = 18
        styles['Result'].spaceAfter = 20
        styles['Result'].textColor = colors.red
        styles['Result'].fontName = "Helvetica-Bold"

    # Add logos
    vivi_logo_path = 'static/WhatsApp Image 2024-06-19 at 6.22.02 PM.jpeg'
    cloud_logo_path = 'static/FAFBF12CEB54491AA27F996EC5D1DF6F.jpg'
    elements.append(RLImage(vivi_logo_path, width=200, height=100))
    # elements.append(RLImage(cloud_logo_path, width=200, height=100))
    elements.append(Spacer(5, 12))

    # Add title
    elements.append(Paragraph("GBM Cancer Detection Report", styles['Centered']))

    # Add Introduction
    elements.append(Paragraph("Introduction", styles['Heading']))
    elements.append(Paragraph("This report provides an analysis of GBM cancer detection and patient outcomes.", styles['BodyText']))

    # Add Patient Data
    elements.append(Paragraph("Patient Data:", styles['Heading']))
    elements.append(Paragraph(f"Fractal Dimension: {fractal_dimension}", styles['BodyText']))
    elements.append(Paragraph(f"Entropy: {entropy}", styles['BodyText']))
    elements.append(Paragraph(f"Lacunarity: {lacunarity}", styles['BodyText']))

    # Add Analysis
    elements.append(Paragraph("Analysis:", styles['Heading']))
    elements.append(Paragraph("The chart below shows the survival duration of GBM patients.", styles['BodyText']))

    # Optionally, you can add the plot image if you have one
    # if file_path:
    #     elements.append(RLImage(file_path, width=500, height=300))

    # Add Conclusion
    elements.append(Paragraph("Conclusion:", styles['Heading']))
    elements.append(Paragraph(f"The analysis indicates the result as: {result}", styles['Result']))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name='GBM_Cancer_Detection_Report.pdf', mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)
