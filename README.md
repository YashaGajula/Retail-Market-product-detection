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
#Results:
![image](https://github.com/user-attachments/assets/576a651e-f3ad-439c-a36d-86693f5159bf)
![image](https://github.com/user-attachments/assets/02999c13-0381-44d7-80a0-409923bbb821)
![image](https://github.com/user-attachments/assets/ccbcd552-dd3a-433e-a188-dac1ee63e173)
![image](https://github.com/user-attachments/assets/c22a3048-b0f8-42b4-bb13-f6d5f676c348)

#Input image:
![image](https://github.com/user-attachments/assets/3fb452a7-b043-4be9-ac8f-ab0da73dc4d1)

#Detected image:
![image](https://github.com/user-attachments/assets/bde98eda-0cb6-4ada-b7b3-ee1dd1016718)

#JSON output for image:
![image](https://github.com/user-attachments/assets/d27446c1-7cd6-4ad4-92b1-51a7c14177e6)

