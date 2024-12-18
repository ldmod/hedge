import pandas as pd
import numpy as np
import h5py


class DataLoader:
    def __init__(self, file_dir, bdays_dir):
        self.file_dir = file_dir
        self.bdays_dir = bdays_dir

        with h5py.File(file_dir) as f:
            self._dates_list = list(f['dates'][:])

        with h5py.File(file_dir) as f:
            self._uid_list = f['stocks'][:]
        try:
            self._uid_list = [i.decode() for i in list(self._uid_list)]
        except:
            pass

    @property
    def dates_list(self):
        return self._dates_list

    @property
    def uid_list(self):
        return self._uid_list

    def get_feature_list(self, end_date):
        try:
            bdays = pd.read_csv(self.bdays_dir)
        except:
            raise ValueError("读取bdays文件错误")
        feature_list = bdays[bdays['bday'] < end_date]['name'].tolist()
        return feature_list

    def field_index_mapping(self, name_list):
        with h5py.File(self.file_dir) as f:
            feature_header = [i.decode() for i in f['feature_header'][:]]
            label_header = [i.decode() for i in f['label_header'][:]]
            status_header = [i.decode() for i in f['status_header'][:]]
        return_res = []
        for name in name_list:
            if name in feature_header:
                return_res.append(('feature', feature_header.index(name)))
            elif name in label_header:
                return_res.append(('label', label_header.index(name)))
            elif name in status_header:
                return_res.append(('status', status_header.index(name)))
            else:
                raise ValueError(f'cannot find field:{name}')
        return return_res

    def dates_index_mapping(self, dates_list):
        '''
        根据传入的日历列表计算每个日期对应的索引
        '''
        dates_list_index = [self.dates_list.index(i) for i in dates_list]
        return dates_list_index

    def get_data(self, name, dates_list):
        '''
        获取数据，在noshfit数据模式下，去除之前的shift逻辑
        '''
        dates_index_list = self.dates_index_mapping(dates_list)
        dates_index_list = [i for i in dates_index_list]
        field, index = self.field_index_mapping([name])[0]
        with h5py.File(self.file_dir) as f:
            data = f[field][dates_index_list, :, index]
            if data.dtype == np.int64:
                data = data.astype(np.float32)
            uid = f['stocks'][:]
            try:
                uid = [i.decode() for i in uid]
            except:
                pass
            data = pd.DataFrame(index=dates_list, columns=uid, data=data)
        return data

    def get_predict_date_list(self, end_date, data_length):
        '''
        根据传入日期，获取长度为data_length的预测日期列表，注意返回日期列表中包含end_date
        '''
        di = self.dates_list.index(end_date)
        return [self.dates_list[i] for i in range(di + 1 - data_length, di + 1) if
                di < self.dates_list.__len__() and i > 0]

    def get_train_date_list(self, end_date, data_length, delay, return_period):
        '''
        根据传入日期，获取长度为data_length的训练日期列表，注意此处不使用end_date当天数据
        '''
        di = self.dates_list.index(end_date)-1
        return [self.dates_list[i] for i in
                range(di - data_length - (delay + 1) - return_period + 2, di - (delay + 1) - return_period + 2)
                if
                di < self.dates_list.__len__() and i > 0]

    def get_cube_data(self, dates_list, feature_list, delay, drop_limit, drop_status,return_name=None, return_period=None,
                      universe=None):
        '''
        获取cube数据，维度分别对应dates * uid * feature，并使用universe过滤
        '''

        field_index_info = self.field_index_mapping(feature_list)
        dates_index_list = self.dates_index_mapping(dates_list)

        field_index_list = [i[1] for i in field_index_info]
        field_list = [i[0] for i in field_index_info]
        if set(field_list).__len__() > 1:
            raise ValueError('feature_list有除特征外其他信息')
        with h5py.File(self.file_dir) as f:
            X = f['feature'][dates_index_list, :][:, :, field_index_list]
        if drop_limit:
            if delay >= 1:
                limit_mask = self.get_data('day_limit', dates_list).values
            else:
                limit_mask = self.get_data('bar_limit', dates_list).values
            limit_mask = np.where(limit_mask == 0, True, False)
            X[~limit_mask] = np.nan
        if drop_status:
            trading_mask = self.get_data("tradingStatus", dates_list).values
            trading_mask = np.where(trading_mask == 2, False, True)
            X[~trading_mask] = np.nan
        X[~np.isfinite(X)] = np.nan
        X[X == 0.] = np.nan
        # 获取universe
        if universe is not None:
            universe = self.get_data(name=universe,
                                     dates_list=dates_list).values.astype(np.bool)
            universe = np.where(universe == 1, True, False)
            X[~universe] = np.nan
        if return_name is not None and return_period is not None:
            # 对y处理
            y = self.get_data(return_name, self.dates_list)
            y = y.rolling(return_period, min_periods=1).sum().shift(-(return_period + (delay + 1) - 1)).reindex(
                dates_list).to_numpy()
            y[~universe] = np.nan
            if drop_limit:
                limit_mask = self.get_data('day_limit', dates_list).values
                limit_mask = np.where(limit_mask == 0, True, False)
                y[~limit_mask] = np.nan
            if drop_status:
                trading_mask = self.get_data("tradingStatus", dates_list).values
                trading_mask = np.where(trading_mask == 2, False, True)
                y[~trading_mask] = np.nan
            return X, y
        else:
            return X, None
