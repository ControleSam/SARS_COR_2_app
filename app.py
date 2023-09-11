import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
from io import StringIO, BytesIO
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import numpy as np
import webbrowser

def molDescrCalc():
    cmd = "java -Xms1G -Xmx1G -Djava.awt.headless=true -jar ./resources/PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./resources/PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./resources -file ./resources/descriptors_output.csv"
    # cmd = 'pushd resources && bash padel.sh && popd'
    process = subprocess.Popen(cmd.split(),stdout=subprocess.PIPE)
    _, _ = process.communicate()
    os.remove('resources/molecule.smi')

def fileDownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

def buildModel(inputData,loadData):

    model = pickle.load(open('resources/sars_coronavirus_2_model.pkl','rb'))
    prediction = model.predict(inputData)
    predictionOutput = pd.Series([np.round(10**(-x)*1e9,2) for x in prediction], name='IC50 [nM]')
    moleculeName = pd.Series(loadData[1],name='molecule_name')
    df = pd.concat([moleculeName, predictionOutput],axis=1)
    df2 = pd.concat([loadData,predictionOutput],axis=1)
    sdf = df.sort_values('IC50 [nM]')
    sdf2 = df2.sort_values('IC50 [nM]')

    return df,sdf,df2,sdf2

def drawSmiles(df):
    mols = [Chem.MolFromSmiles(s) for s in df.iloc[:,0]]
    df[df.columns[-1]] = df[df.columns[-1]].apply(lambda x: str(x)+' nM ')

    opts = Draw.MolDrawOptions()
    opts.legendFraction = 0.2
    opts.legendFontSize = 20
    image = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200,200),legends=[str(x) for x in df[df.columns[-2:]].agg('\n'.join, axis=1)],drawOptions=opts)

    return Image.open(BytesIO(image.data))

def gotoURL(URL):
    webbrowser.open_new_tab(URL)

def prepSession():
    if not 'started' in st.session_state:
        st.session_state.started = True
        st.session_state.data = None
        st.session_state.data_loaded = False
        st.session_state.mol_descr = False
        st.session_state.URL = None
        st.session_state.atDict = {'ChEMBL':'https://www.ebi.ac.uk/chembl/compound_report_card/','NIH':'https://moleculartargets.ccdi.cancer.gov/drug/'}
        # st.session_state.src = None
        # st.session_state.mol = None
        st.session_state.desc_subset = None
        st.session_state.model_built = False
        st.session_state.df = None
        st.session_state.df2 = None
        st.session_state.sdf = None
        st.session_state.sdf2 = None
        st.session_state.smilesImages = None
        st.session_state.available_input = False
        # st.session_state.input1 = ""
        # st.session_state.input2 = None

    st.markdown('''
                # Bioactivity Prediction App for SARS Coronavirus 2

                This is an app that runs on an engine that is trained with ChEMBL bioactivity data for the SARS Coronavirus 2, utilising the [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) [[Read the Paper]](https://doi.org/10.1002/jcc.21707). Given the limitaions in training the engine, the predicted bioactivity results of this app are **not** to be considered suitable for research towards inhibiting the SARS Coronavirus 2.
                
                ---
                ''')
                # **Credits**
                # - App built with `python` + `streamlit`, by Samuel Ntim
                # - Descriptor calculated using [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) [[Read the Paper]](https://doi.org/10.1002/jcc.21707).


    with st.sidebar.header('1. Upload your smiles data'):
        inputString = st.sidebar.text_area("Type (paste) smiles+ChEMBL ID here: ", value="",key='input1',on_change=redo)
        st.sidebar.markdown("**-OR-**")
        uploadedFile = st.sidebar.file_uploader("Upload your input file (multiple smiles)", type=['txt','csv'],key='input2',on_change=redo)
        st.sidebar.markdown("""
    [Example input file](https://raw.githubusercontent.com/ControleSam/SARS_COR_2_app/main/resources/example.csv)
    """)
    
    st.session_state.available_input = (st.session_state.input1 != "" or st.session_state.input2 != None)

    if not st.session_state.available_input:
        st.info(':arrow_left: :runner: You may start by submitting your molecules in the side bar. The format is one line per molecule with canonical smiles in first column and ChEMBL ID in second column, separated by a comma or a whitespaces. See [this example input](https://raw.githubusercontent.com/ControleSam/SARS_COR_2_app/main/resources/example.csv) for the format.')
    predictBtn = st.sidebar.button('Predict',disabled = not st.session_state.available_input)
    if not st.session_state.get('predictBtn'):
        st.session_state.predictBtn = predictBtn

def redo():
    st.session_state.model_built = False
    st.session_state.predictBtn = False

def changeURL():
    st.session_state.URL = st.session_state.atDict[st.session_state.src]+st.session_state.mol

def loadData():
    try:
        data1 = pd.read_table(StringIO(st.session_state.input1), sep='[,*\s+,*]+', header=None, engine='python')
    except:
        data1 = pd.DataFrame()
    try:
        data2 = pd.read_table(st.session_state.input2, sep='[,*\s+,*]+', header=None,engine='python')
    except:
        data2 = pd.DataFrame()

    data = pd.concat([data1,data2],ignore_index=True)
    st.session_state.data = data.drop_duplicates()
    st.session_state.data_loaded = True
    st.session_state.data.to_csv('resources/molecule.smi', sep='\t', header = False, index = False)

def descriptor():
    with st.spinner("Calculating descriptors..."):
        molDescrCalc()
    desc = pd.read_csv('resources/descriptors_output.csv')
    st.session_state.desc = desc # rm
    Xlist = list(pd.read_csv('resources/descriptor_list.csv').columns)
    st.session_state.desc_subset = desc[Xlist]
    st.session_state.mol_descr = True

def model():
    if not st.session_state.model_built:
        st.session_state.df,st.session_state.sdf,st.session_state.df2,st.session_state.sdf2 = buildModel(st.session_state.desc_subset,st.session_state.data)
        st.session_state.smilesImages = drawSmiles(st.session_state.sdf2)
        st.session_state.model_built = True

def molFinder():
    col1, _, col2 = st.columns([2,1,2])
    with col1:
        mol = st.selectbox('Inform me on: ',
                        st.session_state.data[1],key='mol',
                        on_change=changeURL
                        )
    with col2:
        src = st.radio('At: ',
                    st.session_state.atDict.keys(),
                    key='src',
                    on_change=changeURL
                    )
    changeURL()
    # gotoURLBtn = st.button('Go')
    st.markdown(f'<a href="{st.session_state.URL}">Go here</a>', unsafe_allow_html=True)
    # if gotoURLBtn:
    #     gotoURL(st.session_state.URL)

def present():
    # if st.session_state.model_built:
    st.header('**Predition output**')
    st.write(st.session_state.df)
    st.subheader('Molecules in increasing IC50')
    st.image(st.session_state.smilesImages)
    st.markdown(fileDownload(st.session_state.df), unsafe_allow_html=True) 

def predict():
    loadData()
    descriptor()
    model()
    st.session_state.predictBtn = False


# ---------- footer-------------
# taken from https://discuss.streamlit.io/t/st-footer/6447
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
    @import url("https://fonts.googleapis.com/css2?family=Poppins:wght@200;300;400;500;600;700;800;900&display=swap");
   
    # MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    #  .stApp { bottom: 25px; font-family:"Poppins", sans-serif;}
    </style>
    """

    style_div = styles(
        # position="-webkit-sticky",
        position="sticky",
        left=0,
        bottom=0,
        # margin=px(0, 0, 0, 0),
        padding=px(10, 0),
        width=percent(100),
        color="white",
        text_align="center",
        height="50px",
        opacity=1,
        background_color='#24262b', #'#457b9d'
    )

    style_hr = styles(
        display="block",
        margin=px(0, 0, "auto", "auto"),
        border_style="solid",
        border_width=px(1),
        border_color='#F0F2F6'
    )

    body = p()
    foot = div(
        style=style_div
    )(
        # hr(
        #     style=style_hr
        # ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "This Bioactivity Prediction App was built with",
        image('https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/files/python-logo-only.svg',
              width=px(25), height=px(25)),
        ' and ',
        image('https://streamlit.io/images/brand/streamlit-mark-color.svg',width=px(25),height=px(25)),
        " by ",
        link("https://github.com/ControleSam", "Samuel Ntim"),
        # br(),
        # link("https://buymeacoffee.com/chrischross", image('https://i.imgur.com/thJhzOO.png')),
    ]
    layout(*myargs)

def main():
    prepSession()
    if st.session_state.predictBtn:
        predict()
    if st.session_state.model_built:
        molFinder() 
        present()

    # 'st.session_state:', st.session_state
    footer()


main()