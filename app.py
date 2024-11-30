import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle as pkl
import base64
import warnings
warnings.filterwarnings('error')
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

st.set_page_config(page_title='Tunisia Electricity Fraud Detection', page_icon='🕵️', layout="wide", initial_sidebar_state="auto")

# Define CSS style to change text color to black
black_text = """
<style>
body {
    color: black;
}
</style>
"""

# Apply the style to your Streamlit app
st.markdown(black_text, unsafe_allow_html=True)



def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

import base64

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
        .stApp {
            background-image: url("data:image/png;base64,%s");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: scroll;
        }
        .stApp::after {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 255, 255, 0.6);
            z-index: -1;
        }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)



set_png_as_page_bg('dwiinshito--lDNCLbQi9g-unsplash (1).jpg')



st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select a section", ["About", "Predictions",'Help'])

# Load your data
data = pd.read_csv('data/agg_train.csv')

# Identify categorical features
cat_features = ['district', 'client_catg', 'region']

# One-hot encode categorical features
encoder = OneHotEncoder()
encoded = encoder.fit_transform(data[cat_features])

# Combine encoded features with numerical features
numerical_features = ['transactions_count', 'consommation_level_1_mean', 'consommation_level_2_mean', 'consommation_level_3_mean', 'consommation_level_4_mean']
X = pd.concat([pd.DataFrame(encoded.toarray()), data[numerical_features]], axis=1)

# Load the saved model using the load_model() function
with open('model.pkl', 'rb') as f:
    fraud_model = pkl.load(f)


#Dummy function for the model prediction
def predict(district, client_catg, region, transactions_count, consommation_level_1_mean, consommation_level_2_mean, consommation_level_3_mean, consommation_level_4_mean):
    # Insert your model prediction code here
    # This is just a dummy function that returns the sum of the input features
    return transactions_count + consommation_level_1_mean + consommation_level_2_mean + consommation_level_3_mean + consommation_level_4_mean

if app_mode == "Predictions":
    st.write('# Input section.')
    st.write('#### Please fill out every section.')
    col1, col2 = st.columns(2)
    with col1:
        district = st.selectbox('District', data['district'].unique())
        client_catg = st.selectbox('Client Category', data['client_catg'].unique())
        region = st.selectbox('Region', data['region'].unique())
        transactions_count = st.number_input('Transactions Count', value=1, step=1)

    with col2:
        consommation_level_1_mean = st.number_input('Consumption Level 1 Mean', value=0.0, step=0.1)
        consommation_level_2_mean = st.number_input('Consumption Level 2 Mean', value=0.0, step=0.1)
        consommation_level_3_mean = st.number_input('Consumption Level 3 Mean', value=0.0, step=0.1)
        consommation_level_4_mean = st.number_input('Consumption Level 4 Mean', value=0.0, step=0.1)



        input_data = pd.DataFrame([[district, client_catg, region, transactions_count, consommation_level_1_mean, consommation_level_2_mean, consommation_level_3_mean, consommation_level_4_mean]], columns=['district', 'client_catg', 'region', 'transactions_count', 'consommation_level_1_mean', 'consommation_level_2_mean', 'consommation_level_3_mean', 'consommation_level_4_mean'])

        encoded_input = encoder.transform(input_data[cat_features])
        input_features = pd.concat([pd.DataFrame(encoded_input.toarray()), input_data[numerical_features]], axis=1)
        prediction = fraud_model.predict(input_features.values)
        # st.sidebar.write("## Prediction Section")
        st.sidebar.write('##### Click here to predict')


        if st.sidebar.button("Predict", key='predict_button', help='Click here to make a prediction',  use_container_width=False):
            with st.spinner("Predicting..."):
                if prediction[0] == 0:
                    st.sidebar.write('**Prediction:** The customer is not engaging in fraudulent activities.')
                else:
                    st.sidebar.write('**Prediction:** The customer is engaging in fraudulent activities!!.')
                st.sidebar.success("Prediction completed!")


elif app_mode == "About":
    # st.header('Detecting Electricity Fraud In Tunisia')
   
    st.header('Project Overview')
    st.write('##### ⚠️This is not the official Tunisian Company of Electricity and Gas app‼️')
    st.markdown("The Tunisian Company of Electricity and Gas (STEG) is a vital public utility company responsible for supplying electricity and gas across Tunisia. However, the company faced a significant setback when it suffered massive financial losses amounting to 200 million Tunisian Dinars. Investigations later revealed that the primary cause of these losses was fraudulent manipulations of meters by some of the company's consumers. The fraudulent activities resulted in discrepancies in electricity and gas consumption measurements, leading to a significant revenue loss for STEG. This unfortunate incident highlights the importance of implementing effective fraud detection and prevention measures to safeguard the financial stability of public utility companies and prevent the exploitation of services by unscrupulous individuals.")
    st.write('### Objective')
    st.markdown('Using the client’s billing history, the aim of the challenge is to detect and recognize clients involved in fraudulent activities.')

elif app_mode == "Help":
     st.markdown('## Help')
     st.markdown("This app is intended to detect fraudulent activities by Tunisian Company of Electricity and Gas customers (STEG). Here's a quick explanation of the input to help you use this app:")
     st.write('###### To use this app, simply follow these steps:')
     st.markdown("- Fill in the input form with the client's information\n"
                 "- Click the **Predict** button to make a prediction\n"
                "- The app will display a message indicating if the client is engaging in fraudulent activities")
     st.write('The following is the meaning of the inputs:')
     st.markdown("- **Region:** The area where the client is located\n"
            "- **District:** The district where the client is located\n"
            "- **Client_catg:** The category of the client\n"
            "- **Transactions Count**: total number of client transactions\n"
            "- **Consumption level 1:** The lowest level of electricity and gas consumption for the client, typically including basic household appliances such as lights and small electronics.\n"
            "- **Consumption level 2:** The second tier of consumption, including larger household appliances such as refrigerators and washing machines.\n"
            "- **Consumption level 3:** The third tier of consumption, including more energy-intensive appliances such as water heaters and electric cookers.\n"
            "- **Consumption level 4:** The highest level of consumption, including very energy-intensive appliances such as swimming pool pumps or large industrial machinery.")
     st.markdown("\nIf you encounter any issues or have any questions, please feel free to reach out to us using the contact information provided.")
     with st.container():
        st.write("### Contact Information")
        st.write("##### Email: siddhantpanda786786@gmail.com")
        st.write("##### Phone: +919599564384")
        st.write("##### Address: Nairobi, Kenya")
        st.write('##### Github: https://github.com/sodp')