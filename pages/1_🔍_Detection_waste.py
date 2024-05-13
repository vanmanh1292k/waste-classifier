# # streamlit run "d:/machineLearning/Machine_Learning/WebSite/Home.py"

# import streamlit as st
# import numpy as np
# import cv2 as cv
# from PIL import Image
# import base64

# st.set_page_config(
#     page_title="Phát Hiện Khuôn Mặt",
#     page_icon="🔍",
# )

# st.subheader('Phát hiện khuôn mặt')
# st.image(Image.open('D:\machineLearning\Machine Learning\WebSite\images\detection_face.png'), width=700)
# FRAME_WINDOW = st.image('WebSite\\utility\\images\\video_notfound.jpg')
# deviceId = 0
# cap = cv.VideoCapture(deviceId)

# def get_base64(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()


# def set_background(png_file):
#     bin_str = get_base64(png_file)
#     page_bg_img = '''
#     <style>
#     .stApp {
#     background-image: url("data:image/png;base64,%s");
#     background-size: cover;
#     }
#     </style>
#     ''' % bin_str
#     st.markdown(page_bg_img, unsafe_allow_html=True)

# set_background('WebSite\\images\\background2.jpg')

# def add_logo(png_file):
#     bin_str = get_base64(png_file)
#     st.markdown(
#         """
#         <style>
#             [data-testid="stSidebarNav"] {
#                 background-image: url("data:image/png;base64,%s");
#                 background-repeat: no-repeat;
#                 padding-top: 120px;
#                 background-position: 20px 20px;
#             }
#             [data-testid="stSidebarNav"]::before {
#                 content: " Nguyen Van Manh IKBO 04 20";
#                 margin-left: 20px;
#                 margin-top: 20px;
#                 font-size: 20px;
#                 position: relative;
#                 top: 100px;
#             }
#         </style>
#         """ % bin_str,
#         unsafe_allow_html=True,
#     )
# add_logo('D:\machineLearning\Machine_Learning\WebSite\images\logo_mirea.png')


# if 'begining' not in st.session_state:
#     st.session_state.begining = True

# start_btn, stop_btn = st.columns(2)
# with start_btn:
#     start_press = st.button('Start')
# with stop_btn:
#     stop_press = st.button('Stop')


# if start_press:
#     st.session_state.begining = False

# if stop_press:
#     FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')
#     cap.release()


# if 'frame_stop' not in st.session_state:
#     frame_stop = cv.imread('D:\\machineLearning\\Machine Learning\\WebSite\\images\\video_notfound.jpg')
#     st.session_state.frame_stop = frame_stop

# if st.session_state.begining == True:
#     cap.release()



# def visualize(input, faces, fps, thickness=2):
#     if faces[1] is not None:
#         for idx, face in enumerate(faces[1]):
#             # print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

#             coords = face[:-1].astype(np.int32)
#             cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
#             cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
#             cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
#             cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
#             cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
#             cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
#     cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# detector = cv.FaceDetectorYN.create(
#     'D:\\machineLearning\\Machine Learning\\WebSite\\utility\\DetectionFace\\face_detection_yunet_2022mar.onnx',
#     "",
#     (320, 320),
#     0.9,
#     0.3,
#     5000
# )

# tm = cv.TickMeter()
# frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# detector.setInputSize([frameWidth, frameHeight])

# while True:
#     hasFrame, frame = cap.read()
#     if not hasFrame:
#         print('No frames grabbed!')
#         break

#     frame = cv.resize(frame, (frameWidth, frameHeight))

#     # Inference
#     tm.start()
#     faces = detector.detect(frame) # faces is a tuple
#     tm.stop()

#     # Draw results on the input image
#     visualize(frame, faces, tm.getFPS())

#     # Visualize results
#     FRAME_WINDOW.image(frame, channels='BGR')
# cv.destroyAllWindows()
