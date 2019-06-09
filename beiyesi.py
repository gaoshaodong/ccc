wordlist=[
    'my name is xxx',
    'you are stupid',
    'my boyfriend is NB',
    'you looks very smart i like you very much'

]

wordlclass=[0,1,1,0]
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
cvlist=cv.fit_transform(wordlist)
print(cv.get_feature_names())
cvec=cvlist.toarray()
print(cvec)


from sklearn.naive_bayes import GaussianNB
mode1=GaussianNB().fit(cvec,wordlclass)
testword=['i love very much']
ts=cv.transform(testword)
res=ts.toarray()
result=mode1.predict(res)
print(result)