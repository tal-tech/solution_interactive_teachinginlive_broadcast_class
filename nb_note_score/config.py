#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import logging


class Config:
    API_TYPE = 0   # 同步
    VERSION = '1.2.0'
    SERVER_PORT = 8009
    BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
    TMP_FOLDER = os.path.join(BASE_FOLDER, 'tmp')
    if not os.path.exists(TMP_FOLDER):
        os.mkdir(TMP_FOLDER)
    DATA_CHANGE_URL = os.environ.get('DATA_CHANGE_URL')
    EUREKA_URL = os.environ.get('EUREKA_URL')
    COUNT_URL = os.environ.get('COUNT_URL')
    APOLLO_URL = os.environ.get('APOLLO_URL')
    APOLLO_NAMESPACE = os.environ.get('APOLLO_NAMESPACE')
    APP_URL = os.environ.get('APP_URL') or '/aiimage/novabell/note-score'
    DEPLOY_ENV = os.environ.get('DEPLOY_ENV') or 'local'

    EUREKA_APP_NAME = 'NOVABELL-NOTE-SCORE-SERVER'
    EUREKA_HOST_NAME = 'novabell-note-score-server'
    APOLLO_ID = 'novabell'
    APOLLO_KAFKA = 'kafka-bootstrap-servers'
    APOLLO_TOPIC = 'image'
    BIZ_TYPE = 'datawork-image'
    OCR_URL = os.environ.get('OCR_URL') or 'http://gateway-godeye-test.facethink.com/aiimage/novabell/ocr?'

    LOG_LEVEL = logging.DEBUG
    if os.environ.get('LOG_LEVEL') == 'INFO':
        LOG_LEVEL = logging.INFO

    TIME_OUT = 2000

    ALARM_CRITICAL = os.environ.get('ALARM_CRITICAL') or "alertcode:910010001, alerturl:/aiimage/novabell/note-score, alertmsg: NovabellWarning 告警"

    ALERM_DATA = "alertcode:910010001, alerturl:/aiimage/novabell/note-score, alertmsg: DataReflow 告警"
    ALERM_OCR = "alertcode:910010001, alerturl:/aiimage/novabell/note-score, alertmsg: OCRError 告警"

