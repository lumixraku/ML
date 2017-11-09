#!/usr/bin/env python
# coding=utf8


from pyutil.hiveserver2 import connect as hive_connect
from pyutil.presto import  connect as presto_connect

import json

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



if __name__ == '__main__':

    sql = "select form_data_id, log_id, behaviors from ad_tetris_form_submit_user_action_daily where log_id in (SELECT component_id from tetris_user_profile_daily where action_type = 'trigger_smart_captcha' and action_value like '11002' and date = '20171104') and date = '20171104'"
    rs = hive_query(sql, False)

    period = []


    for rec in rs:
        form_data_id = rec[0]
        log_id = rec[1]
        auaclist =  rec[2] #.get('behaviors')

        ltm = 0
        first_focus = 0
        top = 0

        for uacstr in auaclist:
            uac = json.loads(uacstr)
            if uac.get('event_type') == 'impression':

                if uac.get('trigger_time', '0') != '0':
                    ltm = long(uac.get('trigger_time'))
                if uac.get('screen_pos', '0') != '0':
                    top = float(uac.get('screen_pos'))

            if uac.get('event_type') == 'field_focus':

                tritime = long(uac.get('trigger_time'))
                if first_focus == 0:
                    first_focus = tritime
                if first_focus > tritime:
                    first_focus = tritime

        # print ltm, submittime, submittime - ltm
        # 存在一些没有 first_focus 的情况
        if ltm != 0 and ltm != 0L:
            if first_focus != 0:
                # print(form_data_id, log_id, auaclist)
                period.append(  ((first_focus - ltm)/1000, top)  )


    # print len(rs).
    print period
