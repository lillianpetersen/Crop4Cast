from celery import Celery
import requests
import time

app = Celery('test_celery_again', broker='redis://localhost:6379/0')

@app.task
def fetch_url(url):
    print 'in function ', url
    time.sleep(5)
#    resp = requests.get(url)
#    print resp.status_code

def func(urls):
    for url in urls:
        fetch_url.apply_async([url])
        #fetch_url.delay(url)

if __name__ == "__main__":
    func(["http://google.com", "https://amazon.in", "https://facebook.com", "https://twitter.com", "https://alexa.com"])
#
#import sys
#import time
#from celery import Celery
#
#celery = Celery('test_celery_again', broker='redis://localhost:6379/0')
#####################
## Function         #
#####################
#@celery.task  
#def tile_function(tile):
#    print 'in function ', tile
#    time.sleep(5)
#    return
#    
#def top_function(letters):
#   for letter in letters:
#      tile_function(letter)
#
#if __name__ == "__main__":
#   top_function(['a','b','c','d','e','f','g','h'])
