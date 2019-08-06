from flask import Flask, request, render_template ,url_for,redirect, jsonify
app = Flask(__name__)
import numpy as np
import pandas as pd
import nltk
import string
from nltk import word_tokenize
import time
import re
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import csv



if __name__ == '__main__':
	app.run(debug=True, port=5000)


stopwords= set(nltk.corpus.stopwords.words('english'))

def div2word(raw_text, remove_stopwords=False):
	#Remove non-letters, but including numbers
	letters_only = re.sub("[^0-9a-zA-Z]", " ", raw_text)
	#Convert to lower case, split into individual words
	words = letters_only.lower().split()
	if remove_stopwords:
		stops = set(stopwords.words("english")) #searching a set is much faster
		meaningful_words = [w for w in words if not w in stops] # Remove stop words
		words = meaningful_words
	return words

def makeFeatureVec(words, model, wv_size):
	featureVec = np.zeros((wv_size,), dtype="float32")#preallocation of numpy array for speed
	nwords = 0
	
	index2word_set = set(model.wv.index2word)
	
	for word in words:
		if word in index2word_set:
			nwords = nwords + 1
			featureVec = np.add(featureVec, model[word])
# Dividing the result by the number of words to get the average
	if nwords == 0:
		nwords = 1
	featureVec = np.divide(featureVec, nwords)
	return featureVec


wv_size=300 #var for the size or the dimension of the word vector

w2v=joblib.load('w2v_fm')#word2vec model
TM=joblib.load('final_trained_model')#trained MLP model


@app.route("/",methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST']) 
def home():
	if request.method == 'POST':
		data=[]
		data1=request.form.get('data1')
		data2=request.form.get('data2')
		d1_1=div2word(data1)
		d2_1=div2word(data2)
		d1_2=makeFeatureVec(d1_1,w2v,wv_size)
		d2_2=makeFeatureVec(d2_1,w2v,wv_size)
		data.append(d1_2)
		data.append(d2_2)
		tox_lvl=[]
		txc=(TM[0].predict_proba(data)[:,1])*100.000
		svr_txc=(TM[1].predict_proba(data)[:,1])*100.000
		obc=(TM[2].predict_proba(data)[:,1])*100.000
		thrt=(TM[3].predict_proba(data)[:,1])*100.000
		inslt=(TM[4].predict_proba(data)[:,1])*100.000
		idnt_ht=(TM[5].predict_proba(data)[:,1])*100.000
		tox1=round(float(txc[0]),2)
		tox2=round(float(txc[1]),2)
		svtox1=round(float(svr_txc[0]),2)
		svtox2=round(float(svr_txc[1]),2)
		obc1=round(float(obc[0]),2)
		obc2=round(float(obc[1]),2)
		thrt1=round(float(thrt[0]),2)
		thrt2=round(float(thrt[1]),2)
		inslt1=round(float(inslt[0]),2)
		inslt2=round(float(inslt[1]),2)
		idnt1=round(float(idnt_ht[0]),2)
		idnt2=round(float(idnt_ht[1]),2)	
		return redirect(url_for('toxicity',tox1=tox1,tox2=tox2,svtox1=svtox1,svtox2=svtox2,obc1=obc1,obc2=obc2,thrt1=thrt1,thrt2=thrt2,inslt1=inslt1,inslt2=inslt2,idnt1=idnt1,idnt2=idnt2))
		#,sev_toxic=svr_txc,oscene=obc,threat=thrt,insult=inslt,identity=idnt_ht

	return '''<form method="POST" action=''>
					<h3>Enter the first live input(comment1):</h3> <input type="text" name="data1" size=150><br>
					<h3>Enter the second live input(comment2):</h3> <input type="text" name="data2" size=150><br>
					<input type="submit" value="Submit"><br>
				</form>'''

@app.route('/toxicity/<tox1>/<tox2>/<svtox1>/<svtox2>/<obc1>/<obc2>/<thrt1>/<thrt2>/<inslt1>/<inslt2>/<idnt1>/<idnt2>',methods=['GET', 'POST'])
def toxicity(tox1,tox2,svtox1,svtox2,obc1,obc2,thrt1,thrt2,inslt1,inslt2,idnt1,idnt2):
    tox1=tox1
    tox2=tox2
    svtox1=svtox1
    svtox2=svtox2
    obc1=obc1
    obc2=obc2
    thrt1=thrt1
    thrt2=thrt2
    inslt1=inslt1
    inslt2=inslt2
    idnt1=idnt1
    idnt2=idnt2
    if (request.method =='POST'):
    	return redirect(url_for('home'))
    return render_template('toxicity.html',tox1=tox1,tox2=tox2,svtox1=svtox1,svtox2=svtox2,obc1=obc1,obc2=obc2,thrt1=thrt1,thrt2=thrt2,inslt1=inslt1,inslt2=inslt2,idnt1=idnt1,idnt2=idnt2)




