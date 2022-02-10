import numpy as np

class Aggregate:
    month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    month_end = np.cumsum(month_lengths)
    month_start = np.hstack((0, month_end[:-1]))

    def __init__(self, month1, month2, step, ignore_month=True, ignore_shorter=True):
        assert month1 < month2 and month1 >= 1 and month2 <= 12

        self.month1 = month1
        self.month2 = month2
        
        self.m_start = self.month_start[month1-1:month2] - self.month_start[month1-1]
        self.m_end = self.month_end[month1-1:month2] - self.month_start[month1-1]
        self.n = self.m_end[-1]

        if ignore_month:
            self.i_start, self.i_end = self.prepare_split(0, self.n, step, ignore_shorter)
            self.months = np.full(len(self.i_start), np.nan)
        else:
            self.i_start = np.empty(0, dtype='int')
            self.i_end = np.empty(0, dtype='int')
            self.months = np.empty(0, dtype='int')
            for (i_month, (i1, i2)) in enumerate(zip(self.m_start, self.m_end)):
                i_start, i_end = self.prepare_split(i1, i2, step, ignore_shorter)                
                self.i_start = np.hstack((self.i_start, i_start))
                self.i_end = np.hstack((self.i_end, i_end))
                self.months = np.hstack((self.months, np.full(len(i_start), i_month+self.month1)))


    def prepare_split(self, i1, i2, step, ignore_shorter):
        t = np.arange(i1, i2+1, step)
        if t[-1] < i2 and not ignore_shorter:
            t = np.hstack((t, i2))

        return t[:-1], t[1:]


    def split(self, x):
        assert len(x) == self.n

        res = [sum(x[i:j]) for (i, j) in zip(self.i_start, self.i_end)]
        if isinstance(x, np.ndarray):
            return np.array(res)
        else:
            return res


    def convert_to_plot(self, x, box_shape=False):
        assert len(x) == len(self.i_start)

        if box_shape:
            # TODO only works for ignore_shorter=True
            i1 = np.hstack((self.i_start[0], alternate(self.i_start[1:]+0.5, self.i_start[1:]+0.5), self.i_end[-1]-1))
            i2 = alternate(x, x)
        else:
            i1 = 0.5*(self.i_start + self.i_end)
            i2 = x
        return i1, i2


def alternate(x1, x2):
    return np.ravel([x1, x2], 'F')