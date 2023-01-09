import os
import pandas as pd
import math
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from natsort import natsorted

stop_words = stopwords.words('english')
stop_words.remove('in')
stop_words.remove('to')
stop_words.remove('where')

files_name = natsorted(os.listdir('files'))

document_of_terms = []
for files in files_name:

    with open(f'files/{files}', 'r') as f:
        document = f.read()
    tokenized_documents = word_tokenize(document)
    terms = []
    for word in tokenized_documents:
        if word not in stop_words:
            terms.append(word)
    document_of_terms.append(terms)
#--------------------------------------------------------------------------------------------------------------------------------
print("____________________________postional index_______________________________________")
documentNo=1
postionalIndex={}
for doc in document_of_terms:
    for postional,term in enumerate(doc):
        if term in postionalIndex:
            postionalIndex[term][0]=postionalIndex[term][0]+1
            if documentNo in postionalIndex[term][1]:
                postionalIndex[term][1][documentNo].append(postional)
            else:
                postionalIndex[term][1][documentNo]=[postional]
        else:
            postionalIndex[term]=[]
            postionalIndex[term].append(1)
            postionalIndex[term].append({})
            postionalIndex[term][1][documentNo]=[postional]
    documentNo+=1
print(postionalIndex)

#--------------------------------------------------------------------------------------------------------------------------------
print("____________________________tf_______________________________________")
all_words=[]
for doc in document_of_terms:
    for word in doc:
        all_words.append(word)

def getTF(doc):
    wordsFound=dict.fromkeys(all_words,0)
    for word in doc:
        wordsFound[word]+=1
    return  wordsFound


termF=pd.DataFrame(getTF(document_of_terms[0]).values(),index=getTF(document_of_terms[0]))

for i in range(1,len(document_of_terms)):
    termF[i]=getTF(document_of_terms[i]).values()
termF.columns=['doc'+str(i) for i in range(1,11)]
print(termF)
#--------------------------------------------------------------------------------------------------------------------------
print("____________________________tfWeight_______________________________________")
def getWTF(x):
    if x>0:
        return math.log(x)+1
    return 0
for i in range(1,len(document_of_terms)+1):
    termF['doc'+str(i)]=termF['doc'+str(i)].apply(getWTF)
print(termF)
#---------------------------------------------------------------------------------------------------------------------------
print("____________________________IDF_______________________________________")
tfd = pd.DataFrame(columns=['freq', 'idf'])
for i in range(len(termF)):

    frequency = termF.iloc[i].values.sum()

    tfd.loc[i, 'freq'] = frequency

    tfd.loc[i, 'idf'] = math.log10(10 / (float(frequency)))

tfd.index = termF.index
print(tfd)
#------------------------------------------------------------------------------------------------------------------------
print("____________________________TF-IDF_______________________________________")
term_freq_inve_doc_freq = termF.multiply(tfd['idf'], axis=0)
print(term_freq_inve_doc_freq)
#------------------------------------------------------------------------------------------------------------------------
print("____________________________Document Length_______________________________________")
document_length = pd.DataFrame()

def get_docs_length(col):
    return np.sqrt(term_freq_inve_doc_freq[col].apply(lambda x: x**2).sum())

for column in term_freq_inve_doc_freq.columns:
    document_length.loc[0, column+'_lenght'] = get_docs_length(column)
print(document_length)
#------------------------------------------------------------------------------------------------------------------------
print("____________________________Normalized tf.idf_______________________________________")
normalized_term_freq_idf = pd.DataFrame()

def get_normalized(col, x):
    try:
        return x / document_length[col+'_lenght'].values[0]
    except:
        return 0

for column in term_freq_inve_doc_freq.columns:
    normalized_term_freq_idf[column] = term_freq_inve_doc_freq[column].apply(lambda x : get_normalized(column, x))


print(normalized_term_freq_idf)
#-------------------------------------------------------------------------------------------------------------------------
print("____________________________Insert Query and found intersection_______________________________________")
query = input("Enter Your Query: ")
def pos(qury):
    lis= [[]for i in range(10)]
    for term in qury.split():
        if term in postionalIndex.keys():
            for key in postionalIndex[term][1].keys():
              if lis[key-1] !=[]:
                if lis[key-1][-1]==postionalIndex[term][1][key][0]-1:
                  lis[key-1].append(postionalIndex[term][1][key][0])
              else:
                lis[key-1].append(postionalIndex[term][1][key][0])
    positions=[]
    for pos, list in enumerate(lis,start=1):
        if len(list)==len(qury.split()):
            positions.append('doc'+str(pos))
    return positions


def returned(query):
    docouments_found = pos(query)
    if docouments_found == []:
        print('not found')
        return "Miss Search"
    queryTable = pd.DataFrame(index=normalized_term_freq_idf.index)
    queryTable['tf'] = [1 if x in query.split() else 0 for x in list(normalized_term_freq_idf.index)]
    queryTable['w-tf'] = queryTable['tf'].apply(lambda x: getWTF(x))
    product = normalized_term_freq_idf.multiply(queryTable['w-tf'], axis=0)
    queryTable['idf'] = tfd['idf'] * queryTable['w-tf']
    queryTable['tf-idf'] = queryTable['tf'] * queryTable['idf']
    queryTable['normalized'] = 0
    for i in range(len(queryTable)):
        queryTable['normalized'].iloc[i] = float(
            queryTable['idf'].iloc[i] / math.sqrt(sum(queryTable['idf'].values ** 2)))
    print(queryTable)
    product2 = product.multiply(queryTable['normalized'], axis=0)
    print(product2)
    score = {}
    for col in docouments_found:
        score[col] = product2[col].sum()
    print(score)
    queryLenght = math.sqrt(sum(x ** 2 for x in queryTable['idf'].loc[query.split()]))
    print(queryLenght)
    productRes = product2[list(score.keys())].loc[query.split()]
    print(productRes)
    productSum = productRes.sum()
    print(productSum)
    finalScore = sorted(score.items(), key=lambda x: x[1], reverse=True)
    for doc in finalScore:
        print(doc[0], end=' ')


 #----------------------------------------------------------------------------------------------------------------------
returned(query)
# queryTable=pd.DataFrame(index=normalized_term_freq_idf.index)
# queryTable['tf']=[1 if x in query.split() else 0 for x in list(normalized_term_freq_idf.index)]
# queryTable['w-tf']=queryTable['tf'].apply(lambda x:getWTF(x))
# product=normalized_term_freq_idf.multiply(queryTable['w-tf'],axis=0)
# queryTable['idf']=tfd['idf']*queryTable['w-tf']
# queryTable['tf-idf']=queryTable['tf']*queryTable['idf']
# queryTable['normalized']=0
# for i in range(len(queryTable)):
#     queryTable['normalized'].iloc[i]=float(queryTable['idf'].iloc[i]/math.sqrt(sum(queryTable['idf'].values**2)))
# print(queryTable)
# product2=product.multiply(queryTable['normalized'],axis=0)
# #print(product2)
# score={}
# for col in product2.columns:
#     if 0 in product2[col].loc[query.split()].values:
#         pass
#     else:
#         score[col]=product2[col].sum()
# print(score)
# queryLenght=math.sqrt(sum(x**2 for x in queryTable['idf'].loc[query.split()]))
# print(queryLenght)
# productRes=product2[list(score.keys())].loc[query.split()]
# print(productRes)
# productSum=productRes.sum()
# print(productSum)
# finalScore=sorted(score.items(),key=lambda x:x[1],reverse=True)
# for doc in finalScore:
#     print(doc[0],end=' ')
