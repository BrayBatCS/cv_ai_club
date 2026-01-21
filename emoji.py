import cv2
import numpy as np

model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

def cartoon_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

emoji = cv2.imread("moai.png", cv2.IMREAD_UNCHANGED) 
crown = cv2.imread("crown.png", cv2.IMREAD_UNCHANGED)
vader = cv2.imread("vader.png", cv2.IMREAD_UNCHANGED)

def overlay_emoji(frame, emoji, x, y, w, h):
    emoji_resized = cv2.resize(emoji, (w, h))


    y1, y2 = max(0, y), min(frame.shape[0], y+h)
    x1, x2 = max(0, x), min(frame.shape[1], x+w)

    overlay_y1 = 0 if y >= 0 else -y
    overlay_x1 = 0 if x >= 0 else -x
    overlay_y2 = overlay_y1 + (y2 - y1)
    overlay_x2 = overlay_x1 + (x2 - x1)

    alpha_s = emoji_resized[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0 
    alpha_l = 1.0 - alpha_s


    for c in range(3):
        frame[y1:y2, x1:x2, c] = (alpha_s * emoji_resized[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c] + alpha_l * frame[y1:y2, x1:x2, c])
    return frame


def overlay_crown(frame, crown, x, y, w, h):
    crown_resized = cv2.resize(crown, (w, h))

    y1, y2 = max(0, y), min(frame.shape[0], y+h)
    x1, x2 = max(0, x), min(frame.shape[1], x+w)
    
    overlay_y1 = 0 if y >= 0 else -y
    overlay_x1 = 0 if x >= 0 else -x
    overlay_y2 = overlay_y1 + (y2 - y1)
    overlay_x2 = overlay_x1 + (x2 - x1)
    
    alpha_s = crown_resized[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(3):
        frame[y1:y2, x1:x2, c] = (alpha_s * crown_resized[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c] + alpha_l * frame[y1:y2, x1:x2, c])
    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    #Cartoon
    frame = cartoon_filter(frame)

    moai_scale = 1.8
    #Emoji
    for (x, y, w, h) in faces:
        w = int(w*1.8)
        h = int(h*1.8)
        x = int(x - int(0.25*w))
        y = int(y - int(0.25*h))
        frame = overlay_emoji(frame, emoji, x, y, w, h)

    #Vader
    # for (x, y, w, h) in faces:
    #   w = int(w*2)
    #   h = int(h*2)
    #   x = int(x - int(0.3*w))
    #   y = int(y - int(0.25*h))
    #   frame = overlay_emoji(frame, vader, x, y, w, h)

    #Crown
    for (x, y, w, h) in faces:
        w = int(w*1.4)
        h = int(crown.shape[0] * (w / crown.shape[1]))
        x = int(x - int(0.25*w))
        y = int(y - int(h))
        frame = overlay_crown(frame, crown, x, y, w, h)
    

    cv2.imshow("Filter", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
