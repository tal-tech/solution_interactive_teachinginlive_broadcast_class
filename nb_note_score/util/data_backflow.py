#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import json
import time
import traceback
import base64
import requests
from kafka import KafkaProducer
from util.utility import err_code
from config import Config
from util.c_apollo import g_apollo
from util.utility import g_logger, alert_msg


def get_internal_url(src_url, req_id):
    # 外网url转内网url
    if not Config.DATA_CHANGE_URL or src_url is None:
        return src_url
    d = {
        "urls": [src_url],
        "requestId": req_id,
        "sendTime": int(round(time.time() * 1000))
    }
    try:
        ret = requests.post(Config.DATA_CHANGE_URL, json=d, timeout=5)
        g_logger.debug("receive internal url:{}".format(ret.text))
        ret_json = ret.json()
        if ret_json['code'] == 2000000:
            dst_url = ret_json['resultBean'][0]['innerUrl']
        else:
            dst_url = src_url
    except Exception as e:
        g_logger.error("{} - error in change url:{}".format(req_id, e))
        dst_url = src_url
    return dst_url


def send_mq(req_id, msg, input_data, req_time, url=None, b64_data=None, output_data=None):
    if not g_apollo:
        g_logger.debug("not in paas, return")
        return
    try:
        kafka_url = g_apollo.get_value(Config.APOLLO_KAFKA, namespace=Config.APOLLO_NAMESPACE)
        kafka_topic = g_apollo.get_value(Config.APOLLO_TOPIC, namespace=Config.APOLLO_NAMESPACE)
        content = b64_data if b64_data else get_internal_url(url, req_id)

        d = {
            "apiType": Config.API_TYPE,
            "bizType": Config.BIZ_TYPE,
            "requestId": req_id,
            "url": Config.APP_URL,
            "responseTime": int(time.time() * 1000),
            "sendTime": int(time.time() * 1000),
            "sourceInfos": [{
                "id": req_id,
                "sourceType": "base64" if b64_data else "url",
                "content": content,
            }],
            "sourceRemark": input_data,
            "data": output_data,
            "version": Config.VERSION,
            "code": err_code[msg][0],
            "msg": msg,
            "errMsg": 'success',
            'errCode': 200,
            "appKey": input_data.get('appKey') if input_data else '',
            "requestTime": req_time,
        }
        d['duration'] = d['responseTime'] - d['requestTime']

        mq_url = kafka_url.split(',')
        producer = KafkaProducer(bootstrap_servers=mq_url, max_request_size=10*1024*1024)
        msg = json.dumps(d)
        g_logger.debug('{} - request time: {}'.format(req_id, d['duration']))
        g_logger.debug('source info:{}  ----  source remark:{} ---- data:{}'.format(len(json.dumps(d.get('sourceInfos'))),
                                                                                   len(json.dumps(d.get('sourceRemark'))),
                                                                                   len(json.dumps(d.get('data')))))
        # g_logger.debug("{} - kafka url:{} - data length:{}".format(req_id, mq_url, len(msg)))
        producer.send(kafka_topic, msg.encode('utf-8'))
        producer.close()
    except Exception as e:
        g_logger.error(Config.ALERM_DATA + " - {} - 数据回流失败:{}".
                       format(req_id, traceback.format_exception(type(e), e, e.__traceback__)))
