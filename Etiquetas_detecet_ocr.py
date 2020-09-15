#import numpy as np
import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
#import pytesseract
from PIL import Image
import datetime

# Video source - can be camera index number given by 'ls /dev/video*
# or can be a video file, e.g. '~/Video.avi'


#pytesseract.pytesseract.tesseract_cmd = (r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# Video source - can be camera index number given by 'ls /dev/video*
# or can be a video file, e.g. '~/Video.avi'

def detect_etiqueta(etiquetas):
    
    # display function to show image on Jupyter
    def display_img(img,cmap=None):
        fig = plt.figure(figsize = (12,12))
        plt.axis(False)
        ax = fig.add_subplot(111)
        ax.imshow(img,cmap)
    
    # Load the COCO class labels in which our YOLO model was trained on
    
    labelsPath = os.path.join("obj.names")
    LABELS = open(labelsPath).read().strip().split("\n")
    
    # The COCO dataset contains 80 different classes
    #LABELS
    
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.join("yolov4-obj_final.weights")
    configPath = os.path.join("yolov4-obj.cfg")
    
    # Loading the neural network framework Darknet (YOLO was created based on this framework)
    net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)
    #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    
    # Create the function which predict the frame input
    def predict(image):
        
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
        (H, W) = image.shape[:2]
        
        # determine only the "ouput" layers name which we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        # construct a blob from the input image and then perform a forward pass of the YOLO object detector, 
        # giving us our bounding boxes and associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        
        #global boxes
        boxes = []
        confidences = []
        classIDs = []
        threshold = 0.90
        
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
    
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                # confidence type=float, default=0.5
                if confidence > threshold:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
    
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
    
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
    
        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)
    
        # ensure at least one detection exists
        b_bxs = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                b_bxs.append([x,y,w,h])
                
                # draw a bounding box rectangle and label on the image
                color = (0,255,0)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}-conf:{:.2f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x +15, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)
        return image, b_bxs    
   
    # Execute prediction on a single image
    #img = cv2.imread(etiquetas)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    etiquetas = cv2.cvtColor(etiquetas, cv2.COLOR_BGR2RGB)
    im, boxs = predict(etiquetas)
    
    display_img(im)
    
    #print(boxs)
    
    def caract_ocr(imagem,boxs):
    
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imagem = Image.fromarray(im)
        #print(type(imagem))
        #imagem = Image.open("/content/WhatsApp Image 2020-09-11 at 01.03.11.jpeg")
        #print(boxs[3][0],boxs[3][1],(boxs[3][0]+boxs[3][2]),(boxs[3][1]+boxs[3][3]))
        
        id_caracteres=[]
        
        for i in range(len(boxs)):
            #print(i)
            #print(boxs[i][0],boxs[i][1],(boxs[i][0]+boxs[i][2]),(boxs[i][1]+boxs[i][3]))
            etiqueta = imagem.crop((boxs[i][0],boxs[i][1],(boxs[i][0]+boxs[i][2]),(boxs[i][1]+boxs[i][3])))
            display_img(etiqueta)
        
            etiqueta_w, etiqueta_h = etiqueta.size
            #0.60 e 0.70 da largura e altura da imagem sao baseados na posição relativa estimada das letras alvo.
            letras = etiqueta.crop((etiqueta_w*0.60,etiqueta_h*0.70,etiqueta_w,etiqueta_h))
            display_img(letras)
            
            from easyocr import Reader
           
            letras = np.array(letras)
            reader = Reader(['en'])
            results = reader.readtext(letras)
            
            #letters =

            if len(results) != 0:   

                id_caracteres.append(results[0][1])
              
        return id_caracteres
    
    words = caract_ocr(im,boxs)
    
    return words

    
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

key = cv2. waitKey(1)

while True:
    try:
        check, frame = cap.read()
        #print(check) #prints true as long as the webcam is running
        #print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            dt = datetime.datetime.now()
            i = str(dt).replace(":","").replace(" ","_")
            cv2.imwrite(filename='pics/saved_img'+i+'.jpg', img=frame)
            cap.release()
            
            etiq_ocr = detect_etiqueta(frame)
            #cv2.waitKey(1650)
            text_file = open("Output.txt", "a")
            text_file.write("\n {} - {}".format(dt,etiq_ocr))
            text_file.close()
            cv2.destroyAllWindows()
            cap = cv2.VideoCapture(0)
            
        elif key == ord('q'):
            print("Turning off camera.")
            cap.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        cap.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break



'''
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, 0)
    #print(type(gray))
    # Display the resulting frame
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    elif cv2.waitKey(1) == ord('s'): # wait for 's' key to save and exit
        dt = datetime.datetime.now()    
        print(dt)
        #etiq_ocr = detect_etiqueta(gray)
        cv2.imwrite('pallet000.png'.format(dt),frame)
        text_file = open("Output.txt", "a")
        text_file.write("\n {} - {}".format(dt,dt))#etiq_ocr, dt))
        text_file.close()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()'''