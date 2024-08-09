import pandas as ps
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.metrics import accuracy_score as a_s, classification_report as c_r, confusion_matrix as c_m
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
csv = "C:\\Users\\Akshaya Ganesh\\Downloads\\spam.csv"
try:
    dr = ps.read_csv(csv, sep=',', header=0, encoding='ISO-8859-1') 
except UnicodeDecodeError:
    dr = pd.read_csv(csv, sep=',', header=0, encoding='latin1')
dr.columns = ['label', 'message', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
dr['label'] = dr['label'].map({'ham': 0, 'spam': 1})
dr = dr[['label', 'message']]

plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=dr) 
plt.xticks(ticks=[0, 1], labels=['Not Spam', 'Spam'])
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.title('Frequency of Spam and Non-Spam Emails')
plt.show()

dr['message_length'] = dr['message'].apply(len)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='message_length', y='label', data=dr, alpha=0.5)
plt.xticks(ticks=np.arange(0, dr['message_length'].max()+1, step=50))
plt.yticks(ticks=[0, 1], labels=['Not Spam', 'Spam'])
plt.xlabel('Message Length')
plt.ylabel('Label')
plt.title('Scatter Plot of Message Length vs. Email Type')
plt.show()

sorted_dr = dr.sort_values(by='message_length')
cumulative_spam = sorted_dr.groupby('message_length')['label'].sum().cumsum()
plt.figure(figsize=(10, 6))
plt.plot(cumulative_spam.index, cumulative_spam.values, marker='o', linestyle='-', color='b')
plt.xlabel('Message Length')
plt.ylabel('Cumulative Spam Count')
plt.title('Line Plot of Cumulative Spam Count by Message Length')
plt.grid(True)
plt.show()

Xtr, Xtst, ytr, ytst = tts(dr['message'], dr['label'], test_size=0.2, random_state=42)
vect = Tfidf(stop_words='english')
Xtr_transformed = vect.fit_transform(Xtr)
Xtst_transformed = vect.transform(Xtst)
proto_type = mnb()
proto_type.fit(Xtr_transformed, ytr)
y_pre = proto_type.predict(Xtst_transformed)
accuracy = a_s(ytst, y_pre)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", c_r(ytst, y_pre))
conf_mat = c_m(ytst, y_pre)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
def predict_spam(text):
    txt_trans = vect.transform([text])
    prediction = proto_type.predict(txt_trans)
    return "Spam" if prediction[0] == 1 else "Not Spam"
