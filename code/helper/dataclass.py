import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

datasets_dir = os.path.dirname(os.path.realpath(__file__)) + '\\datasets'

# Dataset class

class HDDDataset:
    """
    Implementation of the dataset class
    """
    def __init__(self, dataframe, to_copy=True, name='UnnamedDataset'):
        assert isinstance(dataframe, pd.DataFrame) and isinstance(name, str)
        self.df = dataframe.copy() if to_copy else dataframe
        self.name = name

    @classmethod
    def read_csv(cls, filename, name=None):
        return cls(pd.read_csv(datasets_dir + '\\' + filename), False, name if name else filename[:-4])

    def to_csv(self, to_overwrite=False, filename=None):
        path = datasets_dir + '\\' + (filename if filename else self.name + '.csv')
        if os.path.isfile(path) and not to_overwrite:
            raise RuntimeError(f'to_csv error - {filename if filename else self.name + ".csv"} already exists')
        self.df.to_csv(path, index=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.df.iloc[item]
        return HDDDataset(self.df.iloc[item])

    def __str__(self):
        return self.name + ':\n' + str(self.df)

    def __copy__(self):
        return HDDDataset(self.df, True, self.name)

    def copy(self):
        return self.__copy__()

    # List of colum
    def column_list(self):
        return self.df.columns.values.tolist()

    # Returns a dataset with reassigned indexes. New indexes: [start_index, start_index + 1, ...]
    def reindex(self, start_index=0):
        assert isinstance(start_index, int)
        return HDDDataset(self.df.set_index(pd.Series(range(start_index, start_index + len(self.df)))), False)

    # Relational projection operation. Returns a data set with columns passed as an argument
    def projection(self, columns):
        if isinstance(columns, str):
            return HDDDataset(pd.DataFrame({columns: self.df[columns]}), False)
        if hasattr(columns, '__len__'):
            if len(columns) == 1:
                return HDDDataset(pd.DataFrame({columns[0]: self.df[columns[0]]}), False)
            return HDDDataset(self.df[columns], False)
        raise RuntimeError('Projection argument type error')

    # Mirrored projection operation. Returns a new dataset in which the columns passed in the argument have been deleted
    def exclude_projection(self, exclude_columns):
        columns = self.column_list()
        if isinstance(exclude_columns, str):
            columns.remove(exclude_columns)
        elif hasattr(exclude_columns, '__len__'):
            for column in exclude_columns:
                columns.remove(column) if column else None
        else:
            raise RuntimeError('Exclude projection argument type error')
        if len(columns) == 1:
            return HDDDataset(pd.dataFrame({columns[0]: self.df[columns[0]]}), False)
        return HDDDataset(self.df[columns], False)

    # The operation of merging datasets (vertical, that is, we add rows of the second to the rows of the first one). Returns the dataset
    # left and right work similarly to the connection in RA
    # If left=True, right=True, then the final dataset will contain only the common columns of the original datasets
    # If left=True, right=False, the final dataset will contain the total columns of the original datasets and the remaining columns of the first dataset
    # If left=False, right=True, then it is the same as in the previous case
    # If left=False, right=False, the final dataset will contain the union of the columns of the original datasets
    def merge(self, other, left=False, right=False):
        assert isinstance(other, HDDDataset) or isinstance(other, pd.DataFrame)
        if isinstance(other, pd.DataFrame):
            other = HDDDataset(other, False)
        first = self.reindex(0)
        second = other.reindex(len(self))
        if left:
            second = second.exclude_projection([column if column not in first.column_list() else None for column in second.column_list()])
        if right:
            first = first.exclude_projection([column if column not in second.column_list() else None for column in first.column_list()])
        return HDDDataset(pd.concat([first.df, second.df]), False)

    # Returns a subdataset for a specific serial number
    def get_data_by_serial_number(self, serial_number):
        assert 'serial_number' in self.column_list()
        return HDDDataset(self.df[self.df['serial_number'] == serial_number], False).exclude_projection(['serial_number', 'model'])

    # Returns true if the value in the column the same for all rows of the dataset
    def is_attribute_constant(self, attribute):
        assert attribute in self.column_list()
        return len(self.df[attribute].unique()) == 1

    # Graph of the dependence of one column on another
    def draw_dependancy(self, firstCol, secondCol):
        assert firstCol in self.column_list() and secondCol in self.column_list()
        x = self.df[firstCol]
        y = self.df[secondCol]
        plt.xlabel(firstCol)
        plt.ylabel(secondCol)
        plt.plot(x, y, 'o')
        plt.show()

    # Correlation Matrix
    def draw_correlation(self):
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm')
        plt.show()

    # Time series for a specific serial number for a specific column
    def draw_time_series(self, serial_number, attribute):
        assert attribute in self.column_list()
        toDraw = self.get_data_by_serial_number(serial_number)
        plt.plot(toDraw.df['date'], toDraw.df[attribute])
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        plt.show()
