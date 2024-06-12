import os
import datetime

date = datetime.datetime.now()

os.mkdir(date.strftime('%d.%m.%y'))
# os.mkdir('test')