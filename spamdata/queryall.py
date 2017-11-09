#!/usr/bin/env python
# coding=utf8


from pyutil.hiveserver2 import connect as hive_connect
from pyutil.presto import  connect as presto_connect

import json
import re
def hive_query(sql, use_hive = False):
    result = []
    try:
        if use_hive == True:
            raise ValueError("Foece to use hive")
        print "use presto"
        with presto_connect('presto') as presto_client:
            with presto_client.cursor() as cursor:
                print "begin excute presto"
                cursor.execute(sql)
                result = cursor.fetchall()
    except Exception as err:
        print err
        print 'use hive'
        try:
            with hive_connect('hive', cluster='haruna_default', username='tiger') as client:
                with client.cursor() as cursor:
                    cursor.execute(sql, configuration={'mapreduce.job.name': 'vdver_remain_luonan',\
                                                       'mapreduce.job.priority': 'NORMAL'})
                    result = cursor.fetchall()
        except Exception as err:
            print err
    finally:
        return result


def _uac_feature(uaclist):
        features = {}

        # 1：页面停留时间
        submit_time = 0
        load_finish_time = -1
        page_stay_time = -1
        screen_pos = 0

        # 4：用户行为个数
        component_count = -1
        component_count = len(uaclist)
        features['component_count'] = component_count


        # 6，7: 添加字段填写时间 电话, name, 首次获取焦点时间
        # phone
        has_phone = 0
        change_time_of_telphone = 0
        telphone_time = -1
        telphone_focus_time = -1

        # captcha
        has_captcha = 0
        captcha_time = -1
        captcha_focus_time = -1
        # captcha_flag = True
        # result_captcha_flag = False

        # name
        name_focus_time = -1
        name_blur_time = -1
        name_time = 0

        #focus
        first_focus_time = -1 #最早的 field_focus 的时间
        first_show_time = -1
        try:
            phone_reg = re.compile('1\d{10}')
            eles = uaclist
            # flag = True
            # result_flag = False


            # focus以第一次输入的时间为准  用户可能反复输入  blur 以最后一次为准
            for ele in eles:
                ele = json.loads(ele)
                if ele.get('event_type') == 'impression':
                    load_finish_time = long(ele.get('load_finish_time') or 0)
                    first_show_time = long(ele.get('trigger_time') or 0)
                    screen_pos = ele.get('screen_pos') or 0


                if ele.get('event_type') == 'field_focus':
                    if first_focus_time == -1:
                        first_focus_time = long(ele.get('trigger_time'))

                    if first_focus_time > long(ele.get('trigger_time')) and long(ele.get('trigger_time')) > 1:
                        first_focus_time = long(long(ele.get('trigger_time')))


                if ele.get("element_type") == "captcha":
                    has_captcha = 1
                    if ele.get('event_type') == 'field_focus' and captcha_focus_time == -1:
                        captcha_focus_time = long(ele.get('trigger_time'))
                    if ele.get('event_type') == 'field_blur':
                        captcha_blur_time = long(ele.get('trigger_time'))
                        telphone_time = captcha_blur_time - captcha_focus_time


                if ele.get('element_type') == 'telphone':
                    if ele.get('event_type') == 'field_focus':
                        change_time_of_telphone +=1


                    if phone_reg.match(ele.get('event_value')):
                        has_phone = 1

                    # 只有第一次才算 focus time focustime
                    if ele.get('event_type') == 'field_focus' and telphone_focus_time == -1:
                        telphone_focus_time = long(ele.get('trigger_time'))
                    if ele.get('event_type') == 'field_blur':
                        telphone_blur_time = long(ele.get('trigger_time'))
                        telphone_time = telphone_blur_time - telphone_focus_time

                if ele.get('element_type') == 'name':
                    # name_focus_time 以第一次为准
                    if ele.get('event_type') == 'field_focus' and name_focus_time == -1:
                        name_focus_time = long(ele.get('trigger_time'))
                    if ele.get('event_type') == 'field_blur':
                        name_blur_time = long(ele.get('trigger_time'))
                        name_time = name_blur_time - name_focus_time


                if ele.get('event_type') == 'submit':
                    submit_time = long(ele.get('trigger_time'))
                    page_stay_time = max(submit_time - load_finish_time, 0)
        except Exception as err:
            print('in submit_form_error [form_submit_check] uac_arrange error. err: %s', err)


        features['load_finish_time'] = load_finish_time
        features['page_stay_time'] = page_stay_time
        features['screen_pos'] = screen_pos

        features['captcha'] = has_captcha
        features['name_time'] = name_time
        features['telphone_time'] = telphone_time
        features['phone_yes_or_no'] = has_phone
        if first_show_time == -1:
            features['first_show_time'] = -1
        else:
            features['first_show_time'] = first_show_time - load_finish_time


        if first_focus_time == -1:
            features['first_focus_time'] = -1
        else:
            features['first_focus_time'] = first_focus_time - load_finish_time

        # 其实没有用上
        # features['change_time_of_telphone'] = change_time_of_telphone
        # 8:表单整体提交时长
        features['submit_time'] = submit_time
        total_submit_time = -1

        if first_focus_time > 0 and submit_time > 0:
            total_submit_time = submit_time - first_focus_time
        features['total_submit_time'] = total_submit_time



        # 添加 form_data_id,user_id等信息


        #上面数据整理完毕  开始判断
        return features


if __name__ == '__main__':
    sql = "select form_data_id, log_id, behaviors from ad_tetris_form_submit_user_action_daily where log_id in (SELECT component_id from tetris_user_profile_daily where action_type = 'trigger_smart_captcha' and action_value like '01002' and date = '20171104') and date = '20171104'"
    rs = hive_query(sql, False)

    features = []


    for rec in rs:
        form_data_id = rec[0]
        log_id = rec[1]
        auaclist =  rec[2] #.get('behaviors')

        ltm = 0
        first_focus = 0
        top = 0

        feature = _uac_feature(auaclist)
        features.append(feature)

    print features
    # print len(rs).
    # print period
