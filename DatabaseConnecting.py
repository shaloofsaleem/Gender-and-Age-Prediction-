import cv2
import numpy as np
import mysql.connector
from keras.models import load_model

# Load the trained gender prediction model
model = load_model('gender/datas/gender_and_age_detection_model.h5')

# Open a connection to the MySQL database
db_connection = mysql.connector.connect(
    host="your_host",
    user="your_username",
    password="your_password",
    database="your_database"
)

# Create a cursor to interact with the database
cursor = db_connection.cursor()

videoCap = cv2.VideoCapture(0)
faceDetect = cv2.CascadeClassifier('gender/datas/haarcascade_frontalface_default.xml')
labels_dict = {0: 'Male', 1: 'Female'}

while True:
    ret, frame = videoCap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        # Store the gender prediction in the database
        sql = "INSERT INTO predictions (gender_label) VALUES (%s)"
        values = (labels_dict[label],)
        cursor.execute(sql, values)
        db_connection.commit()

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 225), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 225), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, h), (50, 50, 225), -1)
        cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (225, 255, 255), 2)

    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Close the cursor, database connection, and video capture
cursor.close()
db_connection.close()
videoCap.release()
cv2.destroyAllWindows()
