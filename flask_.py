#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import flask
from flask import Flask
app = Flask(__name__, template_folder='templates')
@app.route('/')
def main():
    return(flask.render_template('main.html'))
if __name__ == '__main__':
    app.run()


# In[ ]:




