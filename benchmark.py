import numpy as np
from time import perf_counter
import numba as nb
from numba import njit, prange
import pandas as pd

def run_strA(df_test=''):
    @njit((nb.int32[:],), cache=True,parallel=True)
    def do_it_numbaParallel(col1):
        res = np.empty(col1.shape[0], dtype='int32')
        for i in prange(col1.shape[0]):
            # print(col1[i])
            if col1[i] > 400:
                res[i] = 0
            elif col1[i] < 400 and col1[i] > 200:
                res[i] = 1
            elif col1[i] < 200 and col1[i] > 0:
                res[i] = 2
            else:
                res[i] = 0

        return res

    @njit((nb.int32[:],), cache=True,parallel=False)
    def do_it_numbaNoParallel(col1):
        res = np.empty(col1.shape[0], dtype='int32')
        for i in range(col1.shape[0]):
            if col1[i] > 400:
                res[i] = 0
            elif col1[i] < 400 and col1[i] > 200:
                res[i] = 1
            elif col1[i] < 200 and col1[i] > 0:
                res[i] = 2
            else:
                res[i] = 0

        return res


    def do_it_non_numba(col1):
        res = np.empty(col1.shape[0], dtype='int32')
        for i in range(col1.shape[0]):
            if col1[i] > 400:
                res[i] = 0
            elif col1[i] < 400 and col1[i] > 200:
                res[i] = 1
            elif col1[i] < 200 and col1[i] > 0:
                res[i] = 2
            else:
                res[i] = 0

        return res


    st = perf_counter()
    df_test['variableA'] = do_it_non_numba(df_test['variableB'].to_numpy())
    en = perf_counter()
    print('Pure Python - Time it took (sec): ',en-st)


    st=perf_counter()
    df_test['variableA'] = np.where(df_test['variableB'] > 400, 'high',
                                         np.where(df_test['variableB'] > 200, 'medium', 'low'))

    en=perf_counter()
    print('npWhere-Time it took (sec): ',en-st)

    st=perf_counter()
    df_test['variableA'] = np.select([df_test['variableB'] > 400, df_test['variableB'] > 200], ['high', 'medium'],default='low')
    en=perf_counter()
    print('npSelect- Time it took (sec): ',en-st)


    st=perf_counter()
    df_test['variableA'] = np.vectorize(lambda x: 'high' if x > 400 else ('medium' if x > 200 else 'low'))(df_test['variableB'])
    en=perf_counter()
    print('np.vectorize - Time it took (sec): ',en-st)


    st=perf_counter()
    df_test['variableA'] = pd.cut(df_test['variableB'], bins=[0, 200, 400, 1000],labels=['low', 'medium', 'high'])
    en=perf_counter()
    print('pd.cut - Time it took (sec): ',en-st)


    st = perf_counter()
    df_test['variableA'] = do_it_numbaParallel(df_test['variableB'].to_numpy())
    en = perf_counter()
    print('Numba Parallel - Time it took (sec): ',en-st)

    st = perf_counter()
    df_test['variableA'] = do_it_numbaNoParallel(df_test['variableB'].to_numpy())
    en = perf_counter()
    print('Numba Non-Parallel - Time it took (sec): ',en-st)





if __name__ == '__main__':
    df_test = pd.DataFrame({'variableB': np.random.randint(0, 1000, 80000000)})
    run_strA(df_test=df_test)
    '''
    Pure Python - Time it took (sec):  16.50152970000636
    npWhere-Time it took (sec):  6.42317819991149
    npSelect- Time it took (sec):  10.901491700089537
    np.vectorize - Time it took (sec):  21.980846599908546
    pd.cut - Time it took (sec):  2.104148200014606
    Numba Parallel - Time it took (sec):  0.10985680005978793
    Numba Non-Parallel - Time it took (sec):  0.3359113000333309
    '''

