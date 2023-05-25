import pickle
import streamlit as st
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix


# Load the trained model
parkinson_model = xgb.XGBClassifier()
parkinson_model=pickle.load(open('/Users/darshan/Desktop/Pdd sys/parktrained_model.sav','rb'))

# Define a function to preprocess the input data
def preprocess_input_data(data):
    # Convert the input data to a Pandas DataFrame
    input_df = pd.DataFrame(data, columns=['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'Jitter(%)', 'Jitter(Abs)',
                                           'RAP', 'PPQ', 'DDP', 'Shimmer', 'Shimmer(dB)', 'APQ3', 'APQ5', 'APQ',
                                           'DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'])

    # Convert the input data to a sparse matrix
    input_sparse = csr_matrix(input_df.values)

    return input_sparse

# Define the Streamlit web app
def main():
    # Define the title and subtitle of the app
    st.title("Parkinson's Disease Prediction")
    st.subheader("Please enter the following information to predict if you have Parkinson's Disease or not:")

    # Get the user input
    fo = st.number_input("MDVP:Fo(Hz)")
    fhi = st.number_input("MDVP:Fhi(Hz)")
    flo = st.number_input("MDVP:Flo(Hz)")
    jitter_percent = st.number_input("Jitter(%)")
    jitter_abs = st.number_input("Jitter(Abs)")
    rap = st.number_input("RAP")
    ppq = st.number_input("PPQ")
    ddp = st.number_input("DDP")
    shimmer = st.number_input("Shimmer")
    shimmer_db = st.number_input("Shimmer(dB)")
    apq3 = st.number_input("APQ3")
    apq5 = st.number_input("APQ5")
    apq = st.number_input("APQ")
    dda = st.number_input("DDA")
    nhr = st.number_input("NHR")
    hnr = st.number_input("HNR")
    rpde = st.number_input("RPDE")
    dfa = st.number_input("DFA")
    spread1 = st.number_input("spread1")
    spread2 = st.number_input("spread2")
    d2 = st.number_input("D2")
    ppe = st.number_input("PPE")

    # Preprocess the input data
    input_sparse = preprocess_input_data([[fo,fhi,flo,jitter_percent,jitter_abs,rap,ppq,ddp,shimmer,shimmer_db,apq3,apq5,apq,dda,nhr,hnr,rpde,dfa,spread1,spread2,d2,ppe]])

    # Predict if the user has Parkinson's Disease or not
    prediction = parkinson_model.predict(input_sparse)

    # Display the prediction
    

    if prediction[0] == 1:
        park_diag=st.subheader("Based on the information provided, it is predicted that a persom have Parkinson's Disease.")
    else:
        park_diag=st.subheader("Based on the information provided, it is predicted that a person do not have Parkinson's Disease.")

# Run the Streamlit app
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    