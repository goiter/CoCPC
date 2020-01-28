

import os
import csv
import numpy as np
from datetime import datetime, timedelta
import random
import pandas as pd
import sklearn
from sklearn.preprocessing import normalize
import io
import pickle


raw_data_path = './data/kdd17/price_long_50/'
preprocess_data_path = './data/kdd17/preprocess/'

class DataLoad():
    def __init__(self):
        self.movement_path = preprocess_data_path
        self.raw_movement_path = raw_data_path
        self.batch_size = 32
        self.train_start_date = '2007-01-01'
        self.train_end_date = '2015-01-01'
        self.valid_start_date = '2015-01-01'
        self.valid_end_date = '2016-01-01'
        self.test_start_date = '2016-01-01'
        self.test_end_date = '2016-12-30'

        self.max_n_days = 20
        self.y_size = 2
        self.shuffle = True
        self.stock_symbols = []

    def _get_stock_symbols(self):
        files = os.listdir(self.raw_movement_path)
        print(len(files))
        stock_symbols = []
        for file in files:
            # print('file:',file)
            stock_symbols.append(file.split('.')[0])
        self.stock_symbols = stock_symbols


    def _get_start_end_date(self, phase):
        if phase == 'train':
            return self.train_start_date, self.train_end_date
        elif phase == 'valid':
            return self.valid_start_date, self.valid_end_date
        elif phase == 'test':
            return self.test_start_date, self.test_end_date
        else:
            return None

    def _get_prices_and_ts(self, ss, main_target_date, valid_date):
        # print('get_prices_and_ts')
        def _get_mv_class(data, use_one_hot=False):
            mv = float(data[1])
            if self.y_size == 2:
                if mv <= 1e-7:
                    return [1.0, 0.0] if use_one_hot else 0
                else:
                    return [0.0, 1.0] if use_one_hot else 1

            if self.y_size == 3:
                threshold_1, threshold_2 = -0.004, 0.005
                if mv < threshold_1:
                    return [1.0, 0.0, 0.0] if use_one_hot else 0
                elif mv < threshold_2:
                    return [0.0, 1.0, 0.0] if use_one_hot else 1
                else:
                    return [0.0, 0.0, 1.0] if use_one_hot else 2


        def _get_prices(data):
            return [float(p) for p in data[2:6]]

        def _get_mv_percents(data):
            return _get_mv_class(data)



        ts, ys, ms, prices, raw_adj_close, mv_percents, mv_class, main_mv_percent = list(), list(), list(), list(), list(), list(), list(), 0.0
        d_t_min = main_target_date - timedelta(days=10)
        stock_movement_path = os.path.join(str(self.movement_path), '{}.txt'.format(ss))
        stock_raw_price_path = os.path.join(str(self.raw_movement_path), '{}.csv'.format(ss))

        flag = 0
        raw_movement_data = pd.read_csv(stock_raw_price_path)

        with io.open(stock_movement_path, 'r', encoding='utf8') as movement_f:
            for i, line in enumerate(movement_f):  # descend
                data = line.split('\t')
                t = datetime.strptime(data[0], '%Y-%m-%d').date()
                # logger.info(t)

                # extract previous prices, mv_percents and current target date

                if t == main_target_date:
                    # logger.info(t)
                    main_mv_percent = data[1]
                    main_market_share = float(data[-1])/1.0e7

                    if -0.005 <= float(main_mv_percent) < 0.0055:  # discard sample with low movement percent
                        return None
                    y = _get_mv_percents(data)
                    main_mv_class = _get_mv_percents(data)
                    flag = 1
                if t in valid_date:
                    ts.append(t)
                    ys.append(_get_mv_percents(data))  #previous mv percent
                    mv_class.append(_get_mv_percents(data))
                    prices.append(_get_prices(data))  # open, high, low, close
                    mv_percents.append(data[1])
                    ms.append(float(data[-1])/1.0e7)

                    revised_raw_date = raw_movement_data.Date.copy()
                    if '/' in raw_movement_data.Date[0]:
                        for k in range(len(raw_movement_data.Date)):
                            revised_raw_date[k] = datetime.strptime(raw_movement_data.Date[k], '%m/%d/%Y').date().strftime('%Y-%m-%d')

                    # print('shape:', raw_movement_data.loc[raw_movement_data.Date==data[0], 'Adj Close'].to_numpy().shape)

                    raw_adj_close.append(raw_movement_data.loc[revised_raw_date==data[0], 'Adj Close'].to_numpy()[0])

                    # print('stock:', ss)
                    # print('t:', data[0])
                    # print('raw adj close price:', raw_adj_close[-1])


        if flag == 0 or len(ts) != len(valid_date):
            return None
        # ascend
        for item in (ts, ys, mv_percents, prices):
            item.reverse()

        prices_and_ts = {
            'ts': ts,
            'ys': ys,
            'y': y,
            'ms': ms,
            'main_mv_class': main_mv_class,
            'mv_class': mv_class,
            'main_mv_percent': main_mv_percent,
            'mv_percents': mv_percents,
            'main_ms_percent': main_market_share,
            'prices': prices,
            'raw_adj_close_prices': raw_adj_close,
        }
        return prices_and_ts


    def sample_gen_from_one_stock(self, s, valid_date, main_target_date):
        """
            generate samples for the given stock.

            => tuple, (x:dict, y_:int, price_seq: list of floats, prediction_date_str:str)
        """

        # logger.info('start _get_prices_and_ts')
        prices_and_ts = self._get_prices_and_ts(s, main_target_date, valid_date)
        if not prices_and_ts:
            # print('none prices and ts')
            return None


        sample_dict = {

            'ms': prices_and_ts['ms'],
            'mv_class': prices_and_ts['mv_class'],
            'main_mv_class': prices_and_ts['main_mv_class'],
            'main_mv_percent': prices_and_ts['main_mv_percent'],
            'mv_percents': prices_and_ts['mv_percents'],
            # target
            'y': prices_and_ts['y'],  # one-hot
            'ys': prices_and_ts['ys'],  # one-hot
            'main_ms_percent': prices_and_ts['main_ms_percent'],
            # source
            'prices': prices_and_ts['prices'],
            'raw_adj_close_prices': prices_and_ts['raw_adj_close_prices'],

        }

        return sample_dict

    def get_valid_time_line(self, target_date_str):
        try_count = 0
        while try_count < 10:
            gen_id = random.randint(0, len(self.stock_symbols) - 1)
            s = self.stock_symbols[gen_id]
            stock_movement_path = os.path.join(str(self.movement_path), '{}.txt'.format(s))
            flag = 0
            step = 0
            valid_dates = []
            # print(stock_movement_path)
            with open(stock_movement_path, 'r') as movement_f:
                for line in movement_f:  # descend
                    data = line.split('\t')
                    _date = datetime.strptime(data[0], '%Y-%m-%d').date()
                    date_str = _date.isoformat()
                    # print('type date_str:', type(date_str))
                    if flag:
                        valid_dates.append(_date)
                        step += 1
                    if step == self.max_n_days:
                        break
                    if date_str == target_date_str:
                        flag = 1
            # print(len(valid_dates))
            if len(valid_dates) == self.max_n_days:
                break
            try_count += 1
        # valid_dates.append(target_date_str)
        valid_dates.reverse()
        return valid_dates



    def batch_gen(self, phase):
        self._get_stock_symbols()
        start_date_str, end_date_str = self._get_start_end_date(phase)

        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        while True:
            rd_time = random.random() * (end_date - start_date) + start_date
            valid_time_line = self.get_valid_time_line(rd_time.isoformat())
            if len(valid_time_line) != 0:
                break


        ys_batch = []
        y_batch = []
        ms_batch = []
        main_mv_class_batch = []
        mv_class_batch = []
        main_mv_percent_batch = []
        main_ms_percent_batch = []
        mv_percent_batch = []
        price_batch = []
        raw_adj_close_price_batch = []
        sample_id = 0
        sampled_stocks = []
        loop_step = 0
        while sample_id < self.batch_size:
            loop_step += 1
            if loop_step == 200:
                break
            # print('sample_id:', sample_id)
            gen_id = random.randint(0, len(self.stock_symbols) - 1)
            s = self.stock_symbols[gen_id]
            if s in sampled_stocks:
                continue

            sample_dict = self.sample_gen_from_one_stock(s, valid_time_line, rd_time)
            # sample_dict = generators[gen_id]
            if not sample_dict:
                continue

            sampled_stocks.append(s)

            # target
            y_batch.append(sample_dict['y'])  # one-hot
            ys_batch.append(sample_dict['ys'])
            ms_batch.append(sample_dict['ms'])
            main_mv_class_batch.append(sample_dict['main_mv_class'])
            mv_class_batch.append(sample_dict['mv_class'])
            main_mv_percent_batch.append(sample_dict['main_mv_percent'])
            mv_percent_batch.append(sample_dict['mv_percents'])
            main_ms_percent_batch.append(sample_dict['main_ms_percent'])
            # source
            price_batch.append(sample_dict['prices'])
            raw_adj_close_price_batch.append(sample_dict['raw_adj_close_prices'])

            sample_id += 1

        if len(y_batch) == self.batch_size:
            batch_dict = {
                # meta
                'main_target_date': rd_time,
                'ts': valid_time_line,
                # target
                'y_batch': y_batch,
                'ys_batch': ys_batch,
                'main_mv_class_batch': main_mv_class_batch,
                'mv_class_batch': mv_class_batch,
                'ms_batch': ms_batch,
                'main_mv_percent_batch': main_mv_percent_batch,
                'mv_percent_batch': mv_percent_batch,
                'main_ms_percent_batch': main_ms_percent_batch,
                # source
                'price_batch': price_batch,
                'raw_adj_close_price_batch': raw_adj_close_price_batch,
            }

            return batch_dict
        else:
            print('batch gen none!')
            return None

    def get_square(self, m):
        return float(m * m)

    def gen_graph(self, data_dict):
        # print('gen_graph')
        batch_size = self.batch_size
        seq_time = len(data_dict['ts'])

        labels = np.zeros((seq_time, batch_size, 1))

        features = np.zeros((seq_time, batch_size, 7))
        raw_adj_close_price_feat = np.zeros((seq_time, batch_size))


        ys = data_dict['ys_batch']
        y = data_dict['y_batch']
        ms = data_dict['ms_batch']
        m = data_dict['main_ms_percent_batch']
        mv_class_batch = data_dict['mv_class_batch']


        # features +ys +ms
        mv_percent_batch = data_dict['mv_percent_batch']
        price_batch = data_dict['price_batch']
        raw_adj_close_price_batch = data_dict['raw_adj_close_price_batch']
        # print('ys')
        for i, e_ys in enumerate(ys):
            re_y = e_ys[1:]
            re_y.append(y[i])
            labels[:, i, :] = np.array(re_y).reshape((seq_time, 1))

        for i in range(batch_size):
            price = np.asarray(price_batch[i]) #T*4
            raw_adj_close_price = np.asarray(raw_adj_close_price_batch[i]) #T*1
            mv_percent = np.asarray(mv_percent_batch[i]).reshape((seq_time, 1))
            ys_ = np.asarray(mv_class_batch[i]).reshape((seq_time, 1))
            ms_ = np.asarray(ms[i]).reshape((seq_time, 1))

            features[:, i, :] = np.hstack((price, mv_percent, ys_, ms_))  #dim:7
            raw_adj_close_price_feat[:,i] = raw_adj_close_price

        return features, labels, raw_adj_close_price_feat, data_dict['ts']

    def _get_dataset(self, phase):
        train_batch_datasets = []
        test_batch_datasets = []
        if phase == 'train':
            print('train data process')
            b = 0
            while b < 200:  # 200
                batch_data = self.batch_gen(phase='train')  # next(train_batch_gen)
                if batch_data is None:
                    print('None batch data')
                    continue

                train_batch_datasets.append(tuple(self.gen_graph(batch_data)))
                b += 1
                print('b:', b)

            print('len dataset:', len(train_batch_datasets))

            with open('./data/kdd17/train_stock_200_batchs.pkl', 'wb') as train_f:
                pickle.dump(train_batch_datasets, train_f)

        elif phase == 'valid':
            print('validation data process')
            # train_batch_gen = self.pipe.batch_gen(phase='train') # a new gen for a new epoch
            b = 0
            while b < 50:
                # for _ in tqdm(range(200)):
                # try:
                batch_data = self.batch_gen(phase='valid')  # next(train_batch_gen)
                if batch_data is None:
                    print('None batch data')
                    continue

                train_batch_datasets.append(tuple(self.gen_graph(batch_data)))
                b += 1
                print('b:', b)
                # except StopIteration:
                #     print('wrong')
            with open('./data/kdd17/valid_stock_50_batchs.pkl', 'wb') as train_f:
                pickle.dump(train_batch_datasets, train_f)

        else:
            print('test data process')
            b = 0
            while b < 50:

                batch_data = self.batch_gen(phase=phase)
                if batch_data is None:
                    continue
                b += 1
                print('b:', b)
                test_batch_datasets.append(tuple(self.gen_graph(batch_data)))
            with open('./data/kdd17/test_stock_50_batchs.pkl', 'wb') as test_f:
                pickle.dump(test_batch_datasets, test_f)

def preprocess():
    '''
    preprocess the raw kdd17 data into unit format
    :return:
    '''
    files = os.listdir(raw_data_path)
    for k, file in enumerate(files):
    # file = files[0]
        with open(os.path.join(raw_data_path,file), 'r') as f:
            data = pd.read_csv(f)
            date_time = data['Date']

            open_price = normalize(data['Open'].to_numpy().reshape(1, -1))
            high_price = normalize(data['High'].to_numpy().reshape(1, -1))
            low_price = normalize(data['Low'].to_numpy().reshape(1, -1))
            close_price = normalize(data['Close'].to_numpy().reshape(1, -1))
            # print('shifted mv:', data['Adj Close'].shift(-1)[:3])

            mv_percent = ((data['Adj Close'] - data['Adj Close'].shift(-1)) / data['Adj Close'].shift(-1))[:-1].to_numpy()
            volume = data['Volume'].to_numpy().reshape(1,-1)

            # print('len date time:', len(date_time))
            # print(date_time[0])
            filename = file.split('.')[0] + '.txt'
            with open(os.path.join(preprocess_data_path, filename), 'a+') as wf:
                for i in range(len(date_time)-1):  #
                    transform_data = []
                    if '/' in date_time[i]:
                        revised_date = datetime.strptime(date_time[i], '%m/%d/%Y').date().strftime('%Y-%m-%d')
                    else:
                        revised_date = date_time[i]
                    transform_data.append(revised_date)
                    # print(mv_percent[i])

                    # transform_data = transform_data.join('\t')
                    transform_data.append(str(round(mv_percent[i], 6)))
                    transform_data.append(str(round(open_price[0, i], 6)))
                    transform_data.append(str(round(high_price[0, i], 6)))
                    transform_data.append(str(round(low_price[0, i], 6)))
                    transform_data.append(str(round(close_price[0, i], 6)))
                    transform_data.append(str(volume[0, i]))

                    wf.write('\t'.join(transform_data))
                    wf.write('\n')



if __name__ == '__main__':

    # preprocess()
    obj = DataLoad()
    obj._get_dataset('train')
    obj._get_dataset('valid')
    obj._get_dataset('test')
    # data_path = './data/kdd17/'
    # filename = 'valid_stock_50_batchs_seqlen_20.pkl'
    # new_batch_loader = []
    # batch_loader = pickle.load(open(os.path.join(data_path, filename), 'rb'))
    # for batch_data in batch_loader:
    #     _, feature, label, _, _, adj_price, time_list = batch_data
    #     new_batch_loader.append((feature,label,adj_price,time_list))
    # with open(os.path.join(data_path, 'valid_stock_50_batchs.pkl'), 'wb') as wf:
    #     pickle.dump(new_batch_loader, wf)




