import tweepy
import json
import csv, os



# Authentication details

access_token = "2919469736-E2BskaQCCyIscNTIcIsp6Y9xWKOw9KwtVLOtaqv"

access_token_secret = "NQ9ZjaUF4rZ4JCxFsl9YwwDMAvociosVwS3XwPmwXWCcE"

consumer_key = "Zv2LA3q2xXRl4VAt4BWWjbMFZ" 

consumer_secret = "XI8NpY5GAZN1NbYIHEsCziemnDMVzvBOx1i3gPMmVXBYH10a6q"


# This is the listener, resposible for receiving data
class StdOutListener(tweepy.StreamListener):
    

    def on_data(self, data):
        tweets = json.loads(data)
        tweets_csv = csv.writer(open("vaccines.csv",mode="a", encoding='utf-8')) # save in csv format

        tweets_text = open("vaccines.txt","a", encoding='utf-8') # save in txt format


        if not os.path.exists('vaccines.json'): 

            json1_file = open('vaccines.json' , 'w', encoding='utf-8')

            json.dump([tweets], json1_file)

        else:

            json1_read = open('vaccines.json' , 'r', encoding='utf-8')

            json_list = json.load(json1_read)

            json_list.append(tweets)

            json1_read.close()

            with open('vaccines.json', 'w+', encoding='utf-8') as my_file:

                json.dump(json_list, my_file)

            

        print("Writing Tweets to file, CTRL-C to terminate")

        print([tweets.get('created_at'),tweets.get('text').encode('ascii', 'ignore'),tweets.get('user').get('screen_name'),tweets.get('source'), tweets.get('user').get('location'), tweets.get('user').get('followers_count'),tweets.get('user').get('friends_count'),tweets.get('retweet_count'),tweets.get('favorite_count'), tweets.get('user').get('lang')])

        print()      

                       

        tweets_csv.writerow([tweets.get('created_at'),                  

                     tweets.get('text').encode('ascii', 'ignore'),                              

                     tweets.get('user').get('screen_name'),

	                 tweets.get('source'),		

                     tweets.get('user').get('location'),        

                     tweets.get('user').get('followers_count'),

                     tweets.get('user').get('friends_count'),

                     tweets.get('user.statuses_count'),

                     tweets.get('user.favourites_count'),

                     tweets.get('user').get('lang')])

        tweets_csv.writerow(" ")

        tweets_text.write(str(tweets.get('text').encode('ascii', 'ignore'))+'\n') 
       
        return True



    def on_error(self, status):

        print (status)



if __name__ == '__main__':

    l = StdOutListener()

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

    auth.set_access_token(access_token, access_token_secret)



    print ("Showing all new tweets")    

    stream = tweepy.Stream(auth, l)

    #stream.filter(track=['programming','java'], languages = "en") 

    #https://dev.twitter.com/rest/reference/get/help/languages

    stream.filter(track=['Vaccines'])
    
