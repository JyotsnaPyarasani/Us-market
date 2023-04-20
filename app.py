import streamlit as st
from pickle import load
import pandas as pd

scaler = load(open('Scalar.pkl',"rb"))
Random_model=load(open('Random_forest_model.pkl','rb'))

st.title(":blue[ US House] :red[Price] Prediction🏠")

print('Enter details:')
Housing_Ratio=st.number_input('Enter Change in Housing_Ratio: ',format = "%.8f")
House_sold=st.number_input('Enter Change in new 1F House sold : ',format = "%.8f")
Construction_permit=st.number_input( 'Enter Change in Current Construction Permits : ',format = "%.8f")
Stock=st.number_input('enter Change in Stock Market Movement: ',format = "%.8f")
Bond=st.number_input('Enter Change in Bond Market Movement: ',format = "%.8f")
rate=st.number_input('enter Change in 10yrs Treasury Rate: ',format = "%.8f")
Expenditure=st.number_input('enter Change in Personal Consumer Expenditure: ',format = "%.8f")
unemployement=st.number_input('enter Change in Long Term Unemployment Rate: ',format = "%.8f")
GDP=st.number_input('enter Change in GDP: ',format = "%.8f")
Year=st.number_input('enter Year: ',format = "%.0f")
Month=st.number_input('enter Month: ',format = "%.0f")

data_us=pd.DataFrame({'H_RATIO_3A_PCT_CHG':Housing_Ratio,'HSN1F_3A_PCT_CHG':House_sold,"PERMIT_3A_PCT_CHG":
                              Construction_permit,"STOCK_MKT_3A_PCT_CHG":Stock,"BAA_YEILD_10Y_2A_PCT_CHG":Bond,"US10Y_3A_PCT_CHG":rate,"RPCE_A_PCT_CHG":
                              Expenditure,"UEMP_3A_PCT_CHG":unemployement,"RGDP_M_PCT_CHG":GDP,"YEAR":Year,"MONTH":Month},index=[0])
data_rescaled = pd.DataFrame(scaler.transform(data_us),
                                     columns = data_us.columns, 
                                    index = data_us.index)

button=st.button("Predict")

if button == True:
   prediction=Random_model.predict(data_rescaled)
   st.success(prediction)
else:
   st.error('Enter values correctly')