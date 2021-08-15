from textblob import TextBlob
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
import re
import string


def resolve_emoticon(line):
   emoticon = {
    	':-)' : 'smile',
        ':-('  : 'sad',
    	':))' : 'very happy',
    	':)'  : 'happy',
    	':((' : 'very sad',
    	':('  : 'sad',
    	':-P' : 'tongue',
    	':-o' : 'gasp',
    	'>:-)':'angry'
   }   
   for key in emoticon:
      line = line.replace(key, emoticon[key])
   return line


def abb_bm(line):
   abbreviation_bm = {
         'sy': 'saya',
         'sk': 'suka',
         'byk': 'banyak',
         'sgt' : 'sangat',
         'mcm' : 'macam',
         'bodo':'bodoh'
   }  
   abbrev = ' '.join (abbreviation_bm.get(word, word) for word in line.split())  
   return (resolve_emoticon(abbrev)) 

  

def abb_en(line):
   abbreviation_en = {
    'u': 'you',
    'thr': 'there',
    'asap': 'as soon as possible',
    'lv' : 'love',    
    'c' : 'see'
   } 
   abbrev = ' '.join (abbreviation_en.get(word, word) for word in line.split())
   return (resolve_emoticon(abbrev))  


def make_plot(pos,neg,neu):
  
   #plots the counts of positive and negative words     

   Polarity = [1,2,3]
   LABELS = ["Positive", "Negative", "Neutral"]
   Count_polarity = [int(pos), int(neg), int(neu)]

   plt.xlabel('Polarity')
   plt.ylabel('Count')
   plt.title('Sentiment Analysis - Lexical Based')

   plt.grid(True)

   plt.bar(Polarity, Count_polarity, align='center')
   plt.xticks(Polarity, LABELS)
   plt.show()



def remove_features(data_str):
       
   url_re = re.compile (r'https?://(\S+)')
   num_re = re.compile (r'(\d+)')
   mention_re = re.compile (r'(@|#)(\w+)')
   RT_re = re.compile (r'RT(\s+)')
   
   data_str = str(data_str)   
   data_str = RT_re.sub (' ', data_str ) # remove RT
   data_str = url_re.sub (' ', data_str ) # remove hyperlinks
   data_str = mention_re.sub (' ', data_str ) # remove @mentions and hash
   data_str = num_re.sub (' ', data_str ) # remove numerical digit

   return data_str

   

def main(sc,filename):

   RDD = sc.textFile(filename).map(lambda x: remove_features(x)).map(lambda x: resolve_emoticon(x)).map(lambda x: x.lower())
   RDD_en = RDD.filter(lambda x: TextBlob(x).detect_language() == 'en').map(lambda x: abb_en(x))
   RDD_ms = RDD.filter(lambda x: TextBlob(x).detect_language() == 'ms').map(lambda x: abb_bm(x)).map(lambda x: TextBlob(str(x)).translate (to = 'en'))
   pos = RDD_en.union(RDD_ms).filter(lambda x: TextBlob(x).sentiment.polarity > 0).count()
   neg = RDD_en.union(RDD_ms).filter(lambda x: TextBlob(x).sentiment.polarity < 0).count()
   neu = RDD_en.union(RDD_ms).filter(lambda x: TextBlob(x).sentiment.polarity == 0).count()
    
   make_plot(int(pos),int(neg),int(neu))
   


if __name__ == "__main__":

   conf = SparkConf().setMaster("local[*]").setAppName("Sentiment Analysis")
   sc = SparkContext(conf=conf)
  
   filename = "vaccines.txt"
  
   main(sc, filename)

   sc.stop()