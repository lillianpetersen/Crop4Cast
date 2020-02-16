from __future__ import absolute_import
from __future__ import absolute_import
from celery import Celery

app = Celery('test_celery_python_code',
             broker='amqp://jimmy:jimmy123@localhost/jimmy_vhost',
             backend='rpc://',
             include=['test_celery.tasks'])

#from test_celery.celery import app
import time


@app.task
def longtime_add(x, y):
    print 'long time task begins'
    # sleep 5 seconds
    time.sleep(5)
    print 'long time task finished'
    return x + y

add=longtime_add(5,6)
