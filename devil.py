import streamlit as st
import py_avataaars as pa
from PIL import Image
import base64
from random import randrange

from time import sleep


from cv2  import *
import numpy as np
import matplotlib.pyplot as plt
import emoji
from tensorflow.keras import models
from tensorflow import keras

# Page title
st.markdown("""
# Avatar Maker

This app allows you to build your own custom avatars based on modular templates provided herein.

**Credits**
- App built in `Python` + `Streamlit` by [Chanin Nantasenamat](https://medium.com/@chanin.nantasenamat) (aka [Data Professor](http://youtube.com/dataprofessor))
- App inspired by the [avataaars generator](https://getavataaars.com) by [Fang-Pen Lin](https://twitter.com/fangpenlin)
- Built with [py_avataaars](https://pypi.org/project/py-avataaars/) library by [Krzysztof Ebert](https://github.com/kebu)
- Avatar is based on Sketch library [Avataaars](https://avataaars.com) designed by [Pablo Stanley](https://twitter.com/pablostanley).
---
""")


list_option_style = ['CIRCLE', 'TRANSPARENT']

list_skin_color = ['TANNED','YELLOW','PALE','LIGHT','BROWN','DARK_BROWN','BLACK']
list_top_type = ['NO_HAIR','EYE_PATCH','HAT','HIJAB','TURBAN',
                 'WINTER_HAT1','WINTER_HAT2','WINTER_HAT3',
                 'WINTER_HAT4','LONG_HAIR_BIG_HAIR','LONG_HAIR_BOB',
                 'LONG_HAIR_BUN','LONG_HAIR_CURLY','LONG_HAIR_CURVY',
                 'LONG_HAIR_DREADS','LONG_HAIR_FRIDA','LONG_HAIR_FRO',
                 'LONG_HAIR_FRO_BAND','LONG_HAIR_NOT_TOO_LONG',
                 'LONG_HAIR_SHAVED_SIDES','LONG_HAIR_MIA_WALLACE',
                 'LONG_HAIR_STRAIGHT','LONG_HAIR_STRAIGHT2',
                 'LONG_HAIR_STRAIGHT_STRAND','SHORT_HAIR_DREADS_01',
                 'SHORT_HAIR_DREADS_02','SHORT_HAIR_FRIZZLE',
                 'SHORT_HAIR_SHAGGY_MULLET','SHORT_HAIR_SHORT_CURLY',
                 'SHORT_HAIR_SHORT_FLAT','SHORT_HAIR_SHORT_ROUND',
                 'SHORT_HAIR_SHORT_WAVED','SHORT_HAIR_SIDES',
                 'SHORT_HAIR_THE_CAESAR','SHORT_HAIR_THE_CAESAR_SIDE_PART']
list_hair_color = ['AUBURN','BLACK','BLONDE','BLONDE_GOLDEN','BROWN',
                   'BROWN_DARK','PASTEL_PINK','PLATINUM','RED','SILVER_GRAY']
list_hat_color = ['BLACK','BLUE_01','BLUE_02','BLUE_03','GRAY_01','GRAY_02',
                  'HEATHER','PASTEL_BLUE','PASTEL_GREEN','PASTEL_ORANGE',
                  'PASTEL_RED','PASTEL_YELLOW','PINK','RED','WHITE']

list_facial_hair_type = ['DEFAULT','BEARD_MEDIUM','BEARD_LIGHT','BEARD_MAJESTIC','MOUSTACHE_FANCY','MOUSTACHE_MAGNUM']
list_facial_hair_color = ['AUBURN','BLACK','BLONDE','BLONDE_GOLDEN','BROWN','BROWN_DARK','PLATINUM','RED']
list_mouth_type = ['DEFAULT','CONCERNED','DISBELIEF','EATING','GRIMACE','SAD','SCREAM_OPEN','SERIOUS','SMILE','TONGUE','TWINKLE','VOMIT']
list_eye_type = ['DEFAULT','CLOSE','CRY','DIZZY','EYE_ROLL','HAPPY','HEARTS','SIDE','SQUINT','SURPRISED','WINK','WINK_WACKY']
list_eyebrow_type = ['DEFAULT','DEFAULT_NATURAL','ANGRY','ANGRY_NATURAL','FLAT_NATURAL','RAISED_EXCITED','RAISED_EXCITED_NATURAL','SAD_CONCERNED','SAD_CONCERNED_NATURAL','UNI_BROW_NATURAL','UP_DOWN','UP_DOWN_NATURAL','FROWN_NATURAL']
list_accessories_type = ['DEFAULT','KURT','PRESCRIPTION_01','PRESCRIPTION_02','ROUND','SUNGLASSES','WAYFARERS']
list_clothe_type = ['BLAZER_SHIRT','BLAZER_SWEATER','COLLAR_SWEATER','GRAPHIC_SHIRT','HOODIE','OVERALL','SHIRT_CREW_NECK','SHIRT_SCOOP_NECK','SHIRT_V_NECK']
list_clothe_color = ['BLACK','BLUE_01','BLUE_02','BLUE_03','GRAY_01','GRAY_02','HEATHER','PASTEL_BLUE','PASTEL_GREEN','PASTEL_ORANGE','PASTEL_RED','PASTEL_YELLOW','PINK','RED','WHITE']
list_clothe_graphic_type = ['BAT','CUMBIA','DEER','DIAMOND','HOLA','PIZZA','RESIST','SELENA','BEAR','SKULL_OUTLINE','SKULL']

if st.button('Random Avatar'):
    index_option_style = randrange(0, len(list_option_style) )
    index_skin_color = randrange(0, len(list_skin_color) )
    index_top_type = randrange(0, len(list_top_type) )
    index_hair_color = randrange(0, len(list_hair_color) )
    index_hat_color = randrange(0, len(list_hat_color) )
    index_facial_hair_type = randrange(0, len(list_facial_hair_type) )
    index_facial_hair_color= randrange(0, len(list_facial_hair_color) )
    index_mouth_type = randrange(0, len(list_mouth_type) )
    index_eye_type = randrange(0, len(list_eye_type) )
    index_eyebrow_type = randrange(0, len(list_eyebrow_type) )
    index_accessories_type = randrange(0, len(list_accessories_type) )
    index_clothe_type = randrange(0, len(list_clothe_type) )
    index_clothe_color = randrange(0, len(list_clothe_color) )
    index_clothe_graphic_type = randrange(0, len(list_clothe_graphic_type) )
else:
    index_option_style = 0
    index_skin_color = 0
    index_top_type = 0
    index_hair_color = 0
    index_hat_color = 0
    index_facial_hair_type = 0
    index_facial_hair_color = 0
    index_mouth_type = 0
    index_eye_type = 0
    index_eyebrow_type = 0
    index_accessories_type = 0
    index_clothe_type = 0
    index_clothe_color = 0
    index_clothe_graphic_type = 0


# Sidebar menu for customizing the avatar

st.sidebar.header('Customize your avatar')

option_style = st.sidebar.selectbox('Style',
                                    list_option_style,
                                    index = index_option_style )

option_skin_color = st.sidebar.selectbox('Skin color',
                                         list_skin_color,
                                         index = index_skin_color )

st.sidebar.subheader('Head top')
option_top_type = st.sidebar.selectbox('Head top',
                                        list_top_type,
                                        index = index_top_type)
option_hair_color = st.sidebar.selectbox('Hair color',
                                         list_hair_color,
                                         index = index_hair_color)
option_hat_color = st.sidebar.selectbox('Hat color',
                                         list_hat_color,
                                         index = index_hat_color)

st.sidebar.subheader('Face')
option_facial_hair_type = st.sidebar.selectbox('Facial hair type',
                                                list_facial_hair_type,
                                                index = index_facial_hair_type)
option_facial_hair_color = st.sidebar.selectbox('Facial hair color',
                                                list_facial_hair_color,
                                                index = index_facial_hair_color)
option_mouth_type = st.sidebar.selectbox('Mouth type',
                                          list_mouth_type,
                                          index = index_mouth_type)
option_eye_type = st.sidebar.selectbox('Eye type',
                                        list_eye_type,
                                        index = index_eye_type)
option_eyebrow_type = st.sidebar.selectbox('Eyebrow type',
                                            list_eyebrow_type,
                                            index = index_eyebrow_type)

st.sidebar.subheader('Clothe and accessories')
option_accessories_type = st.sidebar.selectbox('Accessories type',
                                                list_accessories_type,
                                                index = index_accessories_type)
option_clothe_type = st.sidebar.selectbox('Clothe type',
                                           list_clothe_type,
                                           index = index_clothe_type)
option_clothe_color = st.sidebar.selectbox('Clothe Color',
                                            list_clothe_color,
                                            index = index_clothe_color)
option_clothe_graphic_type = st.sidebar.selectbox('Clothe graphic type',
                                                   list_clothe_graphic_type,
                                                   index = index_clothe_graphic_type)

# Creating the Avatar
# options provided in https://github.com/kebu/py-avataaars/blob/master/py_avataaars/__init__.py
avatar = pa.PyAvataaar(
    #style=pa.AvatarStyle.CIRCLE,
    style=eval('pa.AvatarStyle.%s' % option_style),
    skin_color=eval('pa.SkinColor.%s' % option_skin_color),
    top_type=eval('pa.TopType.SHORT_HAIR_SHORT_FLAT.%s' % option_top_type),
    hair_color=eval('pa.HairColor.%s' % option_hair_color),
    hat_color=eval('pa.ClotheColor.%s' % option_hat_color),
    facial_hair_type=eval('pa.FacialHairType.%s' % option_facial_hair_type),
    facial_hair_color=eval('pa.FacialHairColor.%s' % option_facial_hair_color),
    mouth_type=eval('pa.MouthType.%s' % option_mouth_type),
    eye_type=eval('pa.EyesType.%s' % option_eye_type),
    eyebrow_type=eval('pa.EyebrowType.%s' % option_eyebrow_type),
    nose_type=pa.NoseType.DEFAULT,
    accessories_type=eval('pa.AccessoriesType.%s' % option_accessories_type),
    clothe_type=eval('pa.ClotheType.%s' % option_clothe_type),
    clothe_color=eval('pa.ClotheColor.%s' % option_clothe_color),
    clothe_graphic_type=eval('pa.ClotheGraphicType.%s' %option_clothe_graphic_type)
)

# Custom function for encoding and downloading avatar image
def imagedownload(filename):
    image_file = open(filename, 'rb')
    b64 = base64.b64encode(image_file.read()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

st.subheader('**Rendered Avatar**')
rendered_avatar = avatar.render_png_file('avatar.png')
image = Image.open('avatar.png')
st.image(image)
st.markdown(imagedownload('avatar.png'), unsafe_allow_html=True)

if st.button('CAMERA'):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 32:
            # SPACE pressed
            cv2.imwrite('img_name.png', frame)
            print("CAPTURD")
            break

    cam.release()
    cv2.destroyAllWindows()

    img=cv2.imread("img_name.png")

    face_classifier = cv2.CascadeClassifier(r'streamlit/haarcascade_frontalface_default.xml')
    classifier =load_model(r'streamlit/model.h5')
    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            labels.append(label)
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    for i in labels:
        if i=='Angry':
            st.write(emoji.emojize(":angry_face:"))
        if i=='Disgust':
            st.write(emoji.emojize(":weary_face:"))
        if i=='Fear':
            st.write(emoji.emojize(":face_screaming_in_fear:"))
        if i=='Happy':
            st.write(emoji.emojize(":grinning_face:"))
        if i=='Neutral':
            st.write(emoji.emojize(":neutral_face:"))
        if i=='Sad':
            st.write(emoji.emojize(":sad_but_relieved_face:"))
        if i=='Surprise':
            st.write(emoji.emojize(":face_screaming_in_fear:"))
