import streamlit as st


html_temp = """
<div style="background-color:yellow;padding:1.5px">
<h1 style="color:black;text-align:center;">Demo Web App </h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)
st.title('This is for a good design')
st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)

st.title('Web App')
st.text('Hello Streamlit')

st.header('This is a header')
st.subheader('This is a subheader')


st.markdown('This is a normal Markdown')
st.markdown('# This is a bold Markdown')
st.markdown('## This is a thin-bold Markdown')
st.markdown('* This is a Markdown with point')
st.markdown('** This is a small bold Markdown **')

st.success('Successful')
st.markdown('`This is a markdown`')
st.info("This is an information")
st.warning('This is a warning')
st.error('This is an error')

st.selectbox('Your Occupation', ['Software Engineer', 'Data Scientist'])

st.selectbox('Where do you work', ('Jakarta','Bandung','Surabaya'))

st.slider('What is your level', 0,40, step = 5)