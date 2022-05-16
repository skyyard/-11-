import matplotlib.pyplot as plt
import  matplotlib
import jieba.analyse
import spacy
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

matplotlib.rc("font",family='YouYuan')

def READTEXT()->str:
    with open("a.txt",'r',encoding='utf-8' ) as f:
        return f.read()
def getMDS(data):
    A = (data * data) / 2 * (-1)
    r, c = np.shape(A)

    H = np.eye(r) - np.dot(np.ones((r, 1)), np.ones((1, r))) / r
    B = np.dot(np.dot(H, A), H)

    eig, feat = np.linalg.eig(B)
    eig = np.diag(eig)

    MDS = np.dot(feat[:, [0, 1]], eig[[0, 1], 0:2] ** (0.5))
    return MDS
    # print(MDS)



text=READTEXT()
keywords=jieba.analyse.extract_tags(text, topK=20, withWeight=False, allowPOS=())
area=[text.count(i)*50 for i in keywords]
keylist=''.join(i for i in keywords)
nlp=spacy.load("zh_core_web_sm")

tokens=nlp(keylist)
#

# t1,t2=tokens[0],tokens[1]
# print(t1.vector)
# print(t1.similarity(t2))

ls=[[0]*20 for i in range(20)]
anti_ls=[[0]*20 for i in range(20)]
for i in range(20):
    for j in range(20):
        ls[i][j]=tokens[i].similarity(tokens[j])
for i in range(20):
    for j in range(20):
        anti_ls[i][j]=math.sqrt(ls[i][i]+ls[j][j]-2*ls[i][j])
anti_ls=np.array(anti_ls)


MDS= getMDS(anti_ls)

plt.figure()
print(MDS)
X,Y=MDS[:,[1]],MDS[:,[0]]

plt.scatter(X,Y,s=area)

for i,lable in enumerate(keywords):
    plt.annotate(lable,(X[i],Y[i]))

plt.show()

