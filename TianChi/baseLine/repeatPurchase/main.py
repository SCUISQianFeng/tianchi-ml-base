# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import copy


# 对用户特征进行统计 对点击、加购、购买、收藏分开统计
def col_cnt_(df_data, columns_list, action_type):
    try:
        data_dict = {}
        col_list = copy.deepcopy(columns_list)
        if action_type != None:
            col_list += ['action_type_path']
        for col in col_list:
            # 一个特征下的值拆分成列表
            data_dict[col] = df_data[col].split(' ')

        path_len = len(data_dict[col])
        data_out = []
        for i_ in range(path_len):
            data_txt = ''
            for col_ in columns_list:
                if data_dict['action_type_path'][i_] == action_type:
                    data_txt += '_' + data_dict[col_][i_]
            data_out.append(data_txt)
        return len(data_out)
    except:
        return -1



def col_nunique_(df_data, columns_list, action_type):
    try:
        data_dict = {}
        col_list = copy.deepcopy(columns_list)
        if action_type != None:
            col_list += ['action_type_path']
        for col in col_list:
            # 一个特征下的值拆分成列表
            data_dict[col] = df_data[col].split(' ')
        path_len = len(data_dict[col])
        data_out = []
        for i_ in range(path_len):
            data_txt = ''
            for col_ in columns_list:
                if data_dict['action_type_path'][i_] == action_type:
                    data_txt += '_' + data_dict[col_][i_]
            data_out.append(data_txt)
        return len(set(data_out))
    except:
        return -1



def user_col_cnt(df_data, columns_list, action_type, name):
    df_data[name] = df_data.apply(lambda x: col_cnt_(x, columns_list, action_type), axis=1)
    return df_data


def user_col_nunique(df_data, columns_list, action_type, name):
    df_data[name] = df_data.apply(lambda x: col_nunique_(x, columns_list, action_type), axis=1)
    return df_data


if __name__ == "__main__":
    # all_data_test = pd.read_csv('all_data_test.csv', sep='\t')
    # all_data_test_1000 = all_data_test.head(1000)
    # all_data_test_1000.to_csv('all_data_test_1000.csv', sep='\t', index=True, header=True)
    all_data_test = pd.read_csv('all_data_test_1000.csv', sep='\t')
    # 总点击次数
    all_data_test = all_data_test.copy()
    all_data_test = user_col_cnt(all_data_test, ['seller_path'], '0', 'user_cnt_0')
    # 加入购物车次数
    # all_data_test = user_col_cnt(all_data_test, ['seller_path'], '1', 'user_cnt_1')
    # 购买次数
    # all_data_test = user_col_cnt(all_data_test, ['seller_path'], '2', 'user_cnt_2')
    # 收藏次数
    # all_data_test = user_col_cnt(all_data_test, ['seller_path'], '3', 'user_cnt_3')
    # 不同店铺数
    # all_data_test = user_col_nunique(all_data_test, ['seller_path', 'item_path'], '0', 'seller_nuique_0')
    # all_data_test[['user_cnt_0', 'user_cnt_1', 'user_cnt_2', 'user_cnt_3', 'seller_nuique_0']].head(1000)
