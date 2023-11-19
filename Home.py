import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
from scipy.spatial import distance as dist
import numpy as np
from algo import exact
from algo import heuristic
import openai
from openai import OpenAI
client = OpenAI(api_key=st.secrets["open_ai"])
import os

# Streamlit app title and description
st.image("logo.jpeg")
st.title("Cathay Package Dimension and Label Detection App")
st.write("This app allows you to access your camera or upload an image file to capture package dimensions and labels.")

# Sidebar to choose image source
st.sidebar.header("Image Source")
image_source = st.sidebar.radio("Select Image Source:", ("Camera", "Upload Image"))
    
cv2_img = None
number_of_box = st.selectbox('number of boxes', [1,2,3,4,5,6,7])

for n in range(number_of_box):
    if image_source == "Camera":
        # Access the camera
        st.subheader("Camera Feed")
        st.write("Please grant permission to access your camera.")
        camera = cv2.VideoCapture(0)
        img_file_buffer = st.camera_input("Take a picture",key=n)
        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            path = 'captured_image' + str(n) + '.png'
            print(path)
            cv2.imwrite(path, cv2_img)
            st.success("Image captured successfully!")
    elif image_source == "Upload Image":
        # Upload image file
        st.subheader("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            st.image(image, channels="BGR")
            cv2.imwrite("uploaded_image.png", image)
            st.success("Image uploaded successfully!")
            
def segment_and_get_dimensions(image_path, mode):
    if mode == "camera":
        image = image_path
    else: 
        image = cv2.imread(image_path)
    image2 = image.copy()
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 255, 10)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    largest_rectangle = None

    for contour in contours:
        # Approximate the contour as a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the polygon has four vertices, it's likely a rectangle
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > largest_area:
                largest_area = area
                largest_rectangle = approx
                
    lines = cv2.HoughLinesP(edged, 1, np.pi/180, 100, minLineLength=500, maxLineGap=250)

    # Extend the lines across the entire image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < abs(x2 - x1):
            cv2.line(image2, (0, y1), (image.shape[1], y2), (0, 0, 255), 2)
        else:
            cv2.line(image2, (x1, 0), (x2, image.shape[0]), (0, 0, 255), 2)
        
    hsv_image = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 255, 10)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_area2 = 0
    largest_rectangle2 = None

    for contour in contours:
        # Approximate the contour as a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # If the polygon has four vertices, it's likely a rectangle
        if len(approx) == 4:
            cv2.drawContours(image2, [approx], 0, (0, 255, 0), 2)

    if largest_rectangle is not None:
        x, y, w, h = cv2.boundingRect(largest_rectangle)
        res_w,res_h = 21/w, 29.7/h # pixels per metric
        # Draw the largest rectangle on the original image
        cv2.drawContours(image, [largest_rectangle], 0, (0, 255, 0), 2)
        return image, res_w, res_h
    else:
        print("No rectangles found.")
   
exact_pack = {}
heuristic_pack = {} 
package_dim = []

def ask_openai_chatbot(data,pack):
        # Use OpenAI's chat completion APrI
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "you have to analyze 3D bin packing from the user data and return which packing would be efficient, here are container dimensions: aap = (318, 224, 163),alf = (318,154,163),ama = (318, 244, 244),amf = (318,163,244),ake = (156,154,163) "},
        {"role": "user", "content": "this is my packaage size:"+str(pack)+"and here is my data:" + str(data)}
    ]
    )

        # Extract and return the assistant's reply from the API response
    return completion.choices[0].message.content

# Image processing
if st.button("Capture Package Dimensions and Labels",key='process'):
    image_path = cv2_img if image_source == "Camera" else "uploaded_image.png"
    if image_path: 
        img, w, h = segment_and_get_dimensions(image_path=image_path,mode="camera")
        st.image(img)
        st.write(w,h)
        
        package_dim = [w,h,w]
        boxes = [(w, h, w), (w, h, w), (w, h, w), (w, h, w)]
        aap = (318, 224, 163)
        alf = (318,154,163)
        ama = (318, 244, 244)
        amf = (318,163,244)
        ake = (156,154,163)
        containers = {'aap':aap,'alf':alf,'ama':ama,'amf':amf,'ake':ake}
        for keys,value in containers.items(): 
            exact_pack[keys] = exact.exact_bin_packing(boxes,value)
        
        for keys,value in containers.items(): 
            heuristic_pack[keys] = heuristic.heuristic_bin_packing(boxes,value)
        
        st.write(ask_openai_chatbot({'exact_pack':exact_pack,'heuristic_pack':heuristic_pack},package_dim))
       

st.write("For Exact packing algorithm")
for keys in exact_pack.keys():
    st.image('assets/'+keys+'.jpeg')
    st.write(exact_pack[keys])
  
st.write("For heuristic_pack packing algorithm")  
for keys in heuristic_pack.keys():
    st.image('assets/'+keys+'.jpeg')
    st.write(heuristic_pack[keys])




# print({'exact_pack':exact_pack,'heuristic_pack':heuristic_pack},package_dim)

        
# Display the processed image
if st.button("Download Processed Image"):
    if image_path:
        st.write("Download your processed image.")
        st.download_button("Download", image_path)

# Clear uploaded/captured images
if st.button("Clear Images"):
    if image_source == "Camera":
        camera.release()
    image_path = "captured_image.png" if image_source == "Camera" else "uploaded_image.png"
    if image_path:
        import os

        os.remove(image_path)
        st.warning("Images have been cleared.")

# Information about the app
st.info("This app captures package dimensions and labels in the uploaded or captured image.")

