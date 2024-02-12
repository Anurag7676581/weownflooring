from flask import Flask, render_template, jsonify, request
from PIL import Image , ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from roboflow import Roboflow
from matplotlib.path import Path
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import requests





app = Flask(__name__)
CORS(app)
load_dotenv()

# Initialize Roboflow
rf = Roboflow(api_key="zpylwepWJURMjyaus8J7")
project = rf.workspace().project("room-lyc7i")
model = project.version(1).model

api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')
cloud_name = os.getenv('CLOUD_NAME')

# Function for Boundary Patch Refinement (BPR)
def refine_boundary(mask, patch_size=3):
    # Convert mask to PIL Image
    mask_image = Image.fromarray(np.uint8(mask * 255), 'L')

    # Apply BPR using Gaussian Blur
    refined_mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=patch_size))

    # Convert the refined mask back to numpy array
    refined_mask = np.array(refined_mask_image) / 255.0

    return refined_mask


def create_mask_from_points(image_size, points):
    y, x = np.mgrid[:image_size[1], :image_size[0]]
    points = np.array(points)
    path = Path(points)
    mask = path.contains_points(np.vstack((x.ravel(), y.ravel())).T)
    mask = mask.reshape((image_size[1], image_size[0]))
    return mask

def process_image(room_image_path, texture_image_path):
    # room_image = Image.open(room_image_path)
    room_image = Image.open(requests.get(room_image_path, stream=True).raw)
    texture_image = Image.open(texture_image_path)

    # JSON data
    json_data = model.predict(room_image_path).json()


    # Extract floor coordinates
    floor_data = json_data['predictions'][0]
    print('room image in func',json_data)
    floor_points = floor_data['points']

    # Create a list of tuples for the points
    floor_coordinates = [(point['x'], point['y']) for point in floor_points]

    # Create mask for the floor
    mask = create_mask_from_points(room_image.size, floor_coordinates)

    

    # Apply Boundary Patch Refinement to the mask
    refined_mask = refine_boundary(mask)

    # Ensure that the refined_mask contains boolean values
    refined_mask_bool = refined_mask >=0.9

    # Get the dimensions of the room image
    room_width, room_height = room_image.size

    #continuous image
    resized_texture = texture_image.resize(room_image.size)
    
    # Repeat the texture to cover the entire room
    # repeated_texture = np.tile(np.array(texture_image),
    # (room_height // texture_image.size[1] + 1, room_width // texture_image.size[0] + 1, 1))
    # repeated_texture = repeated_texture[:room_height, :room_width, :]
    # # Use the mask to combine the images


    room_with_texture = np.array(room_image)
    room_with_texture[refined_mask_bool] = np.array(resized_texture)[refined_mask_bool]
    
    #Repeated texture code
    # room_with_texture[mask] = repeated_texture[mask]

    # Convert back to an image
    room_with_texture_image = Image.fromarray(room_with_texture)

    # Save the result
    output_image_path = 'static/newfinal.jpg'  # Save in the static folder to serve via Flask
    room_with_texture_image.save(output_image_path)

    return output_image_path

@app.route('/texture', methods=['POST'])
@cross_origin()
def index():
        cloudinary.config(cloud_name=cloud_name, api_key=api_key, api_secret=api_secret)
        img_url= request.json.get('imgurl')
        texture_idx= request.json.get('texture')
        i = int(texture_idx) #typecasting
        texture_collection = ['t1.jpg', 't2.jpg', 't3.jpg', 't4.jpg']

        texture_image_path = 'uploads/' + texture_collection[i]  
        # Use the locally stored image in the process_image function
        output_image_path = process_image(img_url, texture_image_path)
        textured_img_upload_result = cloudinary.uploader.upload(output_image_path)
        return jsonify({
             "data":textured_img_upload_result
        })
@app.route('/rotateTexture', methods=['POST'])
def solve():
    cloudinary.config(cloud_name=cloud_name, api_key=api_key, api_secret=api_secret)
    img_url= request.json.get('imgurl')
    texture_idx= request.json.get('texture')
    i = int(texture_idx) #typecasting
    texture_collection = ['t1.jpg', 't2.jpg', 't3.jpg', 't4.jpg']

    texture_image_path = 'uploads/' + texture_collection[i]  
        # Use the locally stored image in the process_image function
    # Rotate the image by 90 degrees
    img = Image.open(texture_image_path)
    rotated_img=img.rotate(90)
# Save the rotated image
    path =  'static/rotated_img.jpg' # Replace with the desired output path
    rotated_img.save(path)
    output_image_path = process_image(img_url, path)
    textured_img_upload_result = cloudinary.uploader.upload(output_image_path)
    return jsonify({
             "data":textured_img_upload_result
        })


@app.route("/upload", methods=['POST'])
def upload_file():
    app.logger.info('in upload route')
    cloudinary.config(cloud_name=cloud_name, api_key=api_key, api_secret=api_secret)
    file_to_upload = request.files['file']

    if not file_to_upload:
        return jsonify({'error': 'No file provided'})
    
    texture_image_path = 'uploads/' + 't10.jpeg'
    
    try:
        # Upload the room image to cloudinary
        room_upload_result = cloudinary.uploader.upload(file_to_upload)

        # Process the image and upload the textured image
        output_image_path = process_image(room_upload_result['secure_url'], texture_image_path)
        print(output_image_path)
        textured_img_upload_result = cloudinary.uploader.upload(output_image_path)

        # Return upload results
        return jsonify({'room_image': room_upload_result, 'textured_image': textured_img_upload_result})
    except Exception as e:
        # Handle upload or processing errors
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
