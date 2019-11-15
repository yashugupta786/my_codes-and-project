__author__ = "Yashu Gupta"
from flask import Flask, jsonify
import json
from flask_cors import CORS, cross_origin
from flask import json
from flask.globals import request
from similarity_match import s4x
from sql_connection import sql_connection
import config
import logging
from logging import Formatter, FileHandler
from flask import abort
from pos_tag_based import comment_category
import traceback
import smtplib
#---------------------------------One time sql connection- and class object creation -----------------
obj_sql=sql_connection()
cursor=obj_sql.sql_connect()
obj1=s4x()
print(cursor)
#-------------------------------------------
app = Flask(__name__)
CORS(app)
#-----------------------------------------
def sendEmail(sub, msg):

    FROM = config.FROM
    TO = config.TO
    CC=config.CC
    # Specify your own SMTP and port
    server = smtplib.SMTP(config.smtp_url, 25)
    server.starttls()
    header = 'To:' + TO + '\n' + 'From:' + FROM + '\n'+ 'Cc:'+CC+ '\n' + 'Subject:' + sub + ' \n'
    message = header + '\n ' + msg + ' \n\n'
    server.sendmail(FROM, TO, message)
    server.quit()

@app.route('/datee', methods=['POST'])
@cross_origin()
def datee():
    if request.method != 'POST':
        return json.dumps({"Status": "ERROR", "DATA": None, "Reason": "Only accept POST request"})
    if not request.headers['Content-Type'] == 'application/json':
        return json.dumps({"Status": "ERROR", "DATA": None, "Reason": "Only  accept Content-Type:application/json"})
    if not request.is_json:
        return json.dumps({"Status": "ERROR", "DATA": None,
                           "Reason": 'Expecting json data in the form {"data":"VALUE"}'})
    data = request.json
    if 'data' not in data:
        return json.dumps({"Status": "ERROR", "DATA": None, "Reason": 'Expecting key as data'})
    try:
        statement = data['data']
    except Exception as e:
        return json.dumps({"Status": "ERROR", "DATA": None,
                           "Reason": 'Failed to parse: "data" should be str'})

    try:
        related_terms = obj1.threaded_env(cursor,statement)
        print("------------------------------------------------------------------------------")
        print(related_terms)
        print("------------------------------------------------------------------------------")
        return json.dumps({"Status": "SUCCESS", "DATA": related_terms, "Reason": ""})

    except Exception as e:

        expTrace = str(traceback.format_exc())
        sendEmail("--------------Python error----", expTrace)
        return json.dumps({"Status": "ERROR", "DATA": [{
                        'score': 0,
                        'doc': "null"}], "Reason": ""})



@app.route('/check', methods=['POST'])
@cross_origin()
def check():
    content = request.get_json()

    if content['comment'] == None:
        app.logger.debug("\t" + "NA" + "\t" + "no comment in request")
        abort(400)
    comment = request.json.get('comment')
    app.logger.debug("\t" + str(content))
    #    try:
    result = comment_category(comment)
    response = jsonify(result)
    app.logger.debug("\t" + str(response.get_data()))
    return response


# ---------------------ONE TIME FUNCTION OF API CALLING----------------------------------------------------
def startAPIs():
    try:
        # my_path="D:\comments_duplication_nine"
        file_handler = FileHandler('./logs/output.log')
        handler = logging.StreamHandler()
        file_handler.setLevel(logging.DEBUG)
        handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(Formatter(
            '%(asctime)s %(levelname)s: %(message)s '
            '[in %(pathname)s:%(lineno)d]'
        ))
        handler.setFormatter(Formatter(
            '%(asctime)s %(levelname)s: %(message)s '
            '[in %(pathname)s:%(lineno)d]'
        ))
        app.logger.addHandler(handler)
        app.logger.addHandler(file_handler)

        app.run(config.api_url, port=(config.port_num), debug=False, threaded=True)
        app.run()
    except Exception as e:
        raise ("APIs not started Exception (startAPIs ) at : " + str(config.api_url) + ":" + str(
            config.port_num) + " due to :" + str(e))


if __name__ == '__main__':
    startAPIs()
