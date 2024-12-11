# Retail Market Product Detection

## Overview
This project uses a YOLOv11-based model to detect and classify grocery store shelf items into 9 different categories: **Beauty Product, Biscuit, Bottle, Chocolate, Disinfectants, Meat, Oil, Products, and Shampoo**. The model was trained using a custom dataset, annotated with images of grocery store shelves, and deployed with Flask for a web-based interface for real-time detection.

## Project Structure
project_folder/
│
├── app.py                   
├── inference.py    
├── detection_all_images.py  
├── instructions.txt       
├── requirements.txt         
├── static/                  
│   ├── images/                            
│
├── templates/             
│   ├── index.html
To run this project locally, follow these steps:

```bash
git clone https://github.com/YashaGajula/Retail-Market-product-detection.git
//Install Dependencies:
pip install -r requirements.txt
//Run the Flask Application:
python app.py
//This will run the application locally, and you can access it at http://127.0.0.1:5000/ in your browser.


