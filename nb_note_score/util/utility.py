#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import base64
import json
import logging
import os
import re
import time

import requests
from flask import Response

from config import Config

err_code = {
    "success": (20000, 200),
    "parameters error": (3005054000, 200),
    "illegal url": (3005054001, 200),
    "illegal size": (3005054002, 200),
    "illegal image type": (3005054003, 200),
    "illegal base64": (3005054004, 200),
    "illegal resolution": (3005054005, 200),
    'download error': (3005055001, 200),
    "internal error": (3005055002, 200),
    "ocr error": (3005055003, 200),
    "model error": (3005055004, 200)
}

g_logger = logging.getLogger(__name__)


def init_log():
    ch = logging.StreamHandler()
    ch.setLevel(Config.LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s [%(thread)d] %(levelname)s %(name)s - %(message)s')
    ch.setFormatter(formatter)
    g_logger.setLevel(Config.LOG_LEVEL)
    g_logger.addHandler(ch)


init_log()


def make_response(msg, score, req_id):
    return Response(json.dumps({"msg": msg, "data": {"score": score}, "code": err_code[msg][0],
                                "requestId": req_id}, ensure_ascii=False),
                    status=err_code[msg][1], content_type='application/json')


def make_com_response(msg, complete_ratio, req_id):
    return Response(json.dumps({"msg": msg, "data": {"complete_ratio": complete_ratio}, "code": err_code[msg][0],
                                "requestId": req_id}, ensure_ascii=False),
                    status=err_code[msg][1], content_type='application/json')


def make_all_response(msg, complete_ratio, score, req_id):
    return Response(json.dumps({"msg": msg, "data": {"complete_ratio": complete_ratio, "score": score},
                                "code": err_code[msg][0], "requestId": req_id}, ensure_ascii=False),
                    status=err_code[msg][1], content_type='application/json')


def make_error_response(msg, req_id):
    return Response(json.dumps({"msg": msg, "data": {}, "code": err_code[msg][0],
                                "requestId": req_id}, ensure_ascii=False),
                    status=err_code[msg][1], content_type='application/json')


ip_pattern = re.compile(
    r'^(?:(?:1[0-9][0-9]\.)|(?:2[0-4][0-9]\.)|(?:25[0-5]\.)|(?:[1-9][0-9]\.)|(?:[0-9]\.)){3}'
    r'(?:(?:1[0-9][0-9])|(?:2[0-4][0-9])|(?:25[0-5])|(?:[1-9][0-9])|(?:[0-9]))$')
url_pattern = re.compile(
    r'^(https?)://[\w\-]+(\.[\w\-]+)+([\w\-.,@?^=%&:/~+#]*[\w\-@?^=%&/~+#])?$'
)


def type_url(x):
    try:
        if not (ip_pattern.match(x) or url_pattern.match(x)):
            return False
    except:
        return False
    return True


def get_url_image(url: str, req_id: str):
    ret = None
    for _ in range(3):
        try:
            ret = requests.get(url)
            break
        except:
            time.sleep(1)
    if ret is None:
        return None
    fn = os.path.join(Config.TMP_FOLDER, req_id)
    with open(fn, 'wb') as f:
        f.write(ret.content)
    return fn


def get_b64_image(b64_data: str, req_id: str):
    try:
        fn = os.path.join(Config.TMP_FOLDER, req_id)
        with open(fn, 'wb') as f:
            f.write(base64.b64decode(b64_data.encode()))
        return fn
    except:
        return None


def alert_msg(msg):
    return g_logger.error("{} - {}".format(Config.ALARM_CRITICAL, msg))


def http_transport(encoded_span):
    # encoding prefix explained in https://github.com/Yelp/py_skywalking#transport
    # body = b"\x0c\x00\x00\x00\x01" + encoded_span
    body = encoded_span
    skywalking_url = "http://10.1.13.147:9411/api/v1/spans"
    # skywalking_url = "http://{host}:{port}/api/v1/spans".format(
    # host=app.config["skywalking_HOST"], port=app.config["skywalking_PORT"])
    headers = {"Content-Type": "application/x-thrift"}

    # You'd probably want to wrap this in a try/except in case POSTing fails
    requests.post(skywalking_url, data=body, headers=headers)
