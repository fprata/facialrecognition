import tensorflow as tf
import sys
import os
import cv2
import math

# speicherorte fuer trainierten graph und labels in train.sh festlegen ##

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# image_path = sys.argv[1]
# angabe in console als argument nach dem aufruf  


#bilddatei readen
#image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# holt labels aus file in array 
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("tf_files/retrained_labels.txt")]
# !! labels befinden sich jeweils in eigenen lines -> keine aenderung in retrain.py noetig -> falsche darstellung im windows editor !!
				   
# graph einlesen, wurde in train.sh -> call retrain.py trainiert
with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
 
    graph_def = tf.GraphDef()	## The graph-graph_def is a saved copy of a TensorFlow graph; objektinitialisierung
    graph_def.ParseFromString(f.read())	#Parse serialized protocol buffer data into variable
    _ = tf.import_graph_def(graph_def, name='')	# import a serialized TensorFlow GraphDef protocol buffer, extract objects in the GraphDef as tf.Tensor
	
	#https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/inception.py ; ab zeile 276

face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.1_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')


with tf.Session() as sess:

    video_capture = cv2.VideoCapture(0) 
    #frameRate = video_capture.get(5) #frame rate
    i = 0
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)     
    maxscore = 0
    maxnodeid = 0
    while True:  # fps._numFrames < 120
        frame = video_capture.read()[1] # get current frame
        frameId = video_capture.get(1) #current frame number
        frameR = cv2.resize(frame, (400, 300))        
        #if (frameId % math.floor(frameRate) == 0):
        i = i + 1

        gray = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            h = int(h*1.75)
            y = int(y - h*.25)

            if y < 0:
                y = 0

            sub_face = frameR[y:y+h, x:x+w]


            cv2.imwrite(filename="screens/alpha.png", img=sub_face); # write frame image to file
            image_data = tf.gfile.FastGFile("screens/alpha.png", 'rb').read() # get this image file
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor, \
                     {'DecodeJpeg/contents:0': image_data})     # analyse the image
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            


            for node_id in top_k:

                human_string = label_lines[node_id]
                score = predictions[0][node_id]

                if score > maxscore:
                    maxscore = score
                    maxnodeid = human_string

            cv2.rectangle(frameR,(x,y),(x+w,y+h),(255,255,200),2)

            legend = "{} {:.0%}".format(maxnodeid, maxscore)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frameR, legend, (x, y+h+20), font, 0.5, (255, 255, 200), 1, cv2.LINE_AA)

            print('%s (score = %.5f)' % (maxnodeid, maxscore))

        cv2.imshow("image", frameR)  # show frame in window
        cv2.waitKey(1)  # wait 1ms -> 0 until key input

    video_capture.release() # handle it nicely
    cv2.destroyAllWindows() # muahahaha