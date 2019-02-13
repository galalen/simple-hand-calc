import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to the trained model file")
ap.add_argument("-f", "--frames", type=int, default=20, help="number of consecutive frames to determine the number")
args = vars(ap.parse_args())

import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input

# loading trained model
print(f"[INFO] loading model: {args['model']}")
model = load_model(args["model"])

classes = [1, 2, 3, 4, 5]

# predict class from an image
def predict(roi):
	roi = cv2.resize(roi, (224, 224))
	img = img_to_array(roi)
	img = np.expand_dims(img, axis=0)
	img = preprocess_input(img)
	preds = model.predict(img)
	return classes[np.argmax(preds)]

# helper function to draw labels
def draw_first_number(frame, x, color=(0, 0, 255)):
	cv2.putText(frame, str(x), (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 8)

def draw_second_number(frame, x, color=(0, 0, 255)):
	cv2.putText(frame, str(x), (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 8)

def draw_sign(frame, x, color=(0, 0, 255)):
	cv2.putText(frame, str(x), (130, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 8)

def draw_equal_sign(frame, x="=", color=(0, 0, 255)):
	cv2.putText(frame, str(x), (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 8)

def draw_sum_number(frame, x, color=(0, 0, 255)):
	cv2.putText(frame, str(x), (280, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 8)


# configure camera settings
width, height = 1080, 720

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


green = (0, 255, 0)
red = (0, 0, 255)

# storing the predicted numbers
x_pred = None
y_pred = None

# controlling the drawing of the numbers
draw_x = False
draw_y = False

# used to check for the number brefore drawing it
CONSEC_FRAMES = args["frames"] 
COUNTER = 0

while 1:
	
	ret, frame = cap.read()

	if not ret:
		break

	resized_frame = cv2.resize(frame, (width, height))
	resized_frame = cv2.rectangle(resized_frame, (50, 100), (350, 500), green, 2)
	roi = resized_frame[100:500, 50:350]
	
	# predict first number 
	# and if the number was predicted for more than 
	# the number of the consecutive frames 
	# then we allow drawing it and use it as the first number

	pred = predict(roi)
	
	if x_pred is not None and not draw_x and pred == x_pred:
		COUNTER += 1
		print("XPred: {}".format(x_pred))
		if COUNTER >= CONSEC_FRAMES:
			draw_x = True # allow draw the first number
			COUNTER = 0
			pred = 0
	elif not draw_x and x_pred != pred:
		x_pred = pred
		print("***********************")
		COUNTER = 0

	# we use the same process used for the first number
	# but we start this process after drawing the first numebr 
	# to ensure not getting the same number wrongly

	if draw_x:
		draw_first_number(resized_frame, x_pred)
		draw_sign(resized_frame, "+")

		pred = predict(roi)
		if y_pred is not None and not draw_y and pred == y_pred:
			COUNTER += 1
			print("YPred: {}".format(x_pred))
			if COUNTER >= CONSEC_FRAMES:
				draw_y = True # allow draw the second number
				COUNTER = 0
		elif not draw_y and y_pred != pred:
			y_pred = pred
			COUNTER = 0

	# if the second number was drawn
	# then we can get and draw 
	# the result of the operation

	if draw_y:
		draw_second_number(resized_frame, y_pred)

		result = x_pred + y_pred
		
		draw_equal_sign(resized_frame)
		draw_sum_number(resized_frame, result)


	cv2.imshow('Calc', resized_frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	# if cv2.waitKey(1) & 0xFF == ord('r'):
	# 	print("[INFO] reset the vars...")
	# 	draw_x = False
	# 	draw_y = False
	# 	x_pred = None
	# 	y_pred = None

cv2.destroyAllWindows()
cap.release()