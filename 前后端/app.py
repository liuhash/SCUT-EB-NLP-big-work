from flask import Flask, current_app, redirect, url_for, request
import json

from api import WritePoemApi

app = Flask(__name__)


@app.route('/', methods=["POST"])
def writePoem():  # put application's code here
    # 接收处理json数据请求
    data = json.loads(request.data)  # 将json字符串转为dict
    key = data['key']
    try:
        result = WritePoemApi(key)
    except:
        result = "您找的字不存在语料库之中,请重试"
    return {
        "poem": result
    }



if __name__ == '__main__':
    app.run(host='0.0.0.0')