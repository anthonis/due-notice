#Imports
import streamlit as st
from IPython.display import display
#import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import posixpath
import pandas as pd
from numpy.random import randn
import re
import seaborn as sns
from datetime import datetime
from io import StringIO
import scipy
from sklearn.utils import resample
#from scipy import scipy
import pickle
import datetime

#import wfdb

"""
# DueNotice
A tool to predict preterm births from surface electromyograms
"""

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

#filename = file_selector()
#st.write('You selected `%s`' % filename)

#input_buffer = st.file_uploader("Upload a file", type=("csv")) 
input_buffer = st.file_uploader("Upload .csv file", type=("csv")) 

#myBytesIOObj.seek(0)
#with open(input_buffer, 'wb') as f:
    #shutil.copyfileobj(myBytesIOObj, f, length=131072)

#flip = os.listdir(input_buffer)
#record = wfdb.rdrecord('/Users/madeleineanthonisen/Documents/Random/wfdb-python/term-preterm-ehg-database-1.0.1/tpehgdb/tpehg546') 
#wfdb.plot_wfdb(record=record, title='Record tpehg546 from Physionet Term-Preterm EHG Database') 
# if input_buffer is not None:
# 	print('file uploaded')
# 	data = pd.read_csv(input_buffer,nrows=2)
# 	print('read')

if input_buffer is not None:
	print('file uploaded')
	#ds = np.loadtxt(input_buffer, delimiter='\t')
	#with open('myfile.ext', 'wb') as f:
    #	f.write(io.myBytesIOObj.getvalue())
	#	io.myBytesIOObj.close()
	data22 = pd.read_csv(input_buffer, delimiter='\t')
	#record = wfdb.rdrecord(input_buffer)
	print('read')
	data22 = data22.drop(['Unnamed: 0'],axis=1)
	#in_array = data22.to_numpy()
	#print(type(in_array))

	with open('model_time', 'rb') as f:
		rf = pickle.load(f)


	preds = rf.predict(data22)

	def print_outcome(prediction):
		if prediction[0]==1:
			return 'Delivery expected at term'
		else:
			return 'Delivery expected prematurely'
	st.write(print_outcome(preds))
	#datetime.datetime.now()
	st.write(datetime.datetime.now())

	input_buffer = None






	# import catch22_C

	# def catch22_all(data):

	# 	features = [
	# 	'DN_HistogramMode_5',
	# 	'DN_HistogramMode_10',
	# 	'CO_f1ecac',
	# 	'CO_FirstMin_ac',
	# 	'CO_HistogramAMI_even_2_5',
	# 	'CO_trev_1_num',
	# 	'MD_hrv_classic_pnn40',
	# 	'SB_BinaryStats_mean_longstretch1',
	# 	'SB_TransitionMatrix_3ac_sumdiagcov',
	# 	'PD_PeriodicityWang_th0_01',
	# 	'CO_Embed2_Dist_tau_d_expfit_meandiff',
	# 	'IN_AutoMutualInfoStats_40_gaussian_fmmi',
	# 	'FC_LocalSimple_mean1_tauresrat',
	# 	'DN_OutlierInclude_p_001_mdrmd',
	# 	'DN_OutlierInclude_n_001_mdrmd',
	# 	'SP_Summaries_welch_rect_area_5_1',
	# 	'SB_BinaryStats_diff_longstretch0',
	# 	'SB_MotifThree_quantile_hh',
	# 	'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
	# 	'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
	# 	'SP_Summaries_welch_rect_centroid',
	# 	'FC_LocalSimple_mean3_stderr'
	# 	]	

	# 	data = list(data)

	# 	featureOut = []
	# 	for f in features:
	# 		featureFun = getattr(catch22_C, f)
	# 		featureOut.append(featureFun(data))

	# 	return {'names': features, 'values': featureOut}

	# def unpack_comments(comments):
	#     '''
	#     function to unpack comments and convert them to features
	#     '''
	#     features = []
	#     features.append(int(comments[1].split(" ")[1]))
	#     features.append(float(comments[2].split(" ")[1]))
	#     features.append(float(comments[3].split(" ")[1]))
	#     features.append((comments[4].split(" ")[1]))
	#     features.append((comments[5].split(" ")[1]))
	#     features.append((comments[6].split(" ")[1]))
	#     features.append((comments[7].split(" ")[1]))
	#     features.append(comments[8].split(" ")[1])
	#     features.append(comments[9].split(" ")[1])
	#     features.append(comments[10].split(" ")[1])
	#     features.append(comments[11].split(" ")[1])
	#     features.append(comments[12].split(" ")[1])
	#     features.append(comments[13].split(" ")[1])
	#     features.append(comments[14].split(" ")[1])
	#     return features
	#     #filename = filename.split(".")[0]

	# #make dataframe to fill
	# #df1 = pd.DataFrame(randn(24000,301))
	# #df2 = pd.DataFrame(randn(24000,301))
	# df = pd.DataFrame(columns='RecID Gestation Rectime Age Parity Abortions Weight Hypertension Diabetes Ppos Bleed1T Bleed2T Funneling Smoker'.split())
	# df2 = pd.DataFrame(columns='c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19 c20 c21 c22'.split())

	# directory_path = '/Users/madeleineanthonisen/Documents/Random/wfdb-python/term-preterm-ehg-database-1.0.1/tpehgdb/'
	# #directory = os.fsencode(directory_in_str)

	# #Loop to apply preprocessing to all files in directory
	# i=0
	# for file in os.listdir(directory_path):
	#     filename = os.fsdecode(file)
	#     #if filename.endswith(".dat") or filename.endswith(".hea"):
	#     if filename.endswith(".dat"):
	#         i+=1
	#         filename = filename.split(".")[0]
	#         #print(filename)
	#         signals, fields = wfdb.rdsamp('/Users/madeleineanthonisen/Documents/Random/wfdb-python/term-preterm-ehg-database-1.0.1/tpehgdb/'+filename, channels=[9,10])
	        
	#         #cut for noise at start&end of recording
	#         signals = signals[6000:30000]
	        
	#         #Get features from fields and put in df
	#         comments = fields['comments']
	#         features = unpack_comments(comments)
	#         s = pd.Series(features, index=df.columns)
	#         df = df.append(s, ignore_index=True)
	        
	#         #use catch22 to get features
	#         #channel 9
	#         #signals[:,0]
	#         s2 = pd.Series(catch22_all(signals[:,0])['values'],index=df2.columns)
	#         df2 = df2.append(s2, ignore_index=True)
	        
	        
	#         #df[str(i)] = signals
	#         # print(os.path.join(directory, filename))
	#     else:
	#          continue
	# #print(features)
	# #print(i)
	# #print(len(signals))
	# #print(signals.shape)

	# dfm2 = df.merge(df2, how='outer', left_index=True, right_index=True)
	# #Create a categorical column off the 'Gestation' to predict
	# def gest_bin(time):
	#     '''function to create a binay column from the 'Gestation' column to separate Preterm (0) from Term (1)
	#     '''
	#     if time<37:
	#         return 0
	#     else:
	#         return 1

	# dfm2['bin'] = dfm2['Gestation'].apply(gest_bin)

	# def get_int(age):
	#     if age != 'None':
	#         age = int(age)
	    
	#     return age

	# dfm2['Age'] = dfm2['Age'].apply(get_int)
	# dfm2['Age'].replace(to_replace=['None'], value=np.nan, inplace=True)

	# def impute_age(cols):
	#     Age = cols[0]
	#     Bin = cols[1]
	    
	#     if pd.isnull(Age):

	#         if Bin == 1:
	#             return 29.4

	#         else:
	#             return 29.3

	#     else:
	#         return Age

	# dfm2['Age'] = dfm2[['Age','bin']].apply(impute_age,axis=1)

	# df_minority = dfm2[dfm2['bin']==0]
	# df_minority_upsampled = resample(df_minority, replace=True,n_samples=262)
	# df_upsampled = pd.concat([dfm2[dfm2['bin']==1], df_minority_upsampled])

	# #dfn = pd.concat([dfm,dfm2])

	# from sklearn.neighbors import KNeighborsClassifier
	# from sklearn.linear_model import LogisticRegression
	# from sklearn.ensemble import RandomForestClassifier
	# from sklearn.naive_bayes import GaussianNB

	# from sklearn.model_selection import train_test_split

	# from sklearn.metrics import confusion_matrix,classification_report,f1_score,accuracy_score

	# # Define a function that runs the 4 algorithms and gives the accuracy, overall (macro) f1 score and confusion matrix for each

	# # def learn(X, y, state):
	# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=state)
	# #     models = [('kNN', KNeighborsClassifier()), ('log_reg', LogisticRegression()), ('bayes', GaussianNB()), ('forest', RandomForestClassifier())]
	    
	# #     for name, classifier in models:
	# #         model = classifier
	# #         pred = model.fit(X_train, y_train).predict(X_test)
	# #         f1 = f1_score(y_test, pred, average='macro')
	# #         acc = accuracy_score(y_test, pred)
	# #         print(name)
	# #         print('f1-score:', f1)
	# #         print('accuracy:', acc)
	# #         print(confusion_matrix(y_test, pred), '\n')
	# #     return f1

	# X = df_upsampled.drop(['RecID', 'Gestation', 'Parity', 'Abortions', 'Weight',
	#        'Hypertension', 'Diabetes', 'Ppos', 'Bleed1T', 'Bleed2T', 'Funneling',
	#        'Smoker','bin'],axis=1)
	# #y = dfm2['bin']
	# y = df_upsampled['bin']

	# model = RandomForestClassifier()
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
	# model.fit(X_train, y_train).predict(X_test)

	# data22

	# prediction = model.predict(data22.iloc[0])

	# def print_outcome(prediction):
	# 	if prediction[0]==1:
	# 		return 'Expected at term'
	# 	else:
	# 		return 'Expected prematurely'

	# #X = dfm2.drop(['RecID', 'Gestation', 'Parity', 'Abortions', 'Weight',
	#   #     'Hypertension', 'Diabetes', 'Ppos', 'Bleed1T', 'Bleed2T', 'Funneling',
	#  #      'Smoker','bin','Rectime','Age'],axis=1)
	# #y = dfm2['bin']

	# st.write(print_outcome(prediction))
