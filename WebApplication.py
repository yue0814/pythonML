# Saving the current state of a trained machine learning model
# Using SQLite databases for data storage
# Developing a Web application  using Flask
# Deploying a ML application to a public Web server

import pickle
import os


dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.mkdir(dest)
