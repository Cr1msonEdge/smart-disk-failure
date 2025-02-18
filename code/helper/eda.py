import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from helper.dataclass import HDDDataset
import matplotlib.dates as mdates
from scipy.stats import skew, kurtosis


class EDA():
    def __init__(self, data: HDDDataset):
        self.data = data
        self.df = data.df
        self.selected_disk_df = None   # Chosen disk
        self.selected_disk_num = None
        self.SMART_LIST = [i for i in self.df.columns if i.startswith('smart_')]

    def get_serial_numbers(self):
        return self.df['serial_number'].unique()

    def select_disk(self, serial_number):
        assert serial_number in self.get_serial_numbers()
        self.selected_disk_df = self.df[self.df['serial_number'] == serial_number]
        self.selected_disk_num = serial_number
    
    def unselect_disk(self):
        self.selected_disk_df = None
        self.selected_disk_num = None
    
    def show_info(self):
        print("=== Overall information ===")
        print(self.df.info())
        print("\n=== First rows of information ===")
        print(self.df.head())
        print("\n=== Descriptional stats ===")
        print(self.df.describe())
        print("\n=== Missing values ===")
        print(self.df.isnull().sum())

    def show_assymetry(self):
        if self.selected_disk_df is not None:
            print(f"=== Skewness for SMART features of chosen disk: {self.selected_disk_num} ===")
            for feature in self.SMART_LIST:
                print(f"{feature}: {skew(self.selected_disk_df[feature])} - {kurtosis(self.selected_disk_df[feature])}")
        else:
            print("=== Skewness for SMART features for the whole dataset ===")
            for feature in self.SMART_LIST:
                print(f"{feature}: {skew(self.df[feature])} - {kurtosis(self.df[feature])}")

        
    def plot_histograms(self, features=None):
        if self.selected_disk_df is not None:
            if features is not None: 
                for i in features: 
                    assert i in self.selected_disk_df.columns
                numeric_columns = features
            else:
                numeric_columns = self.selected_disk_df.select_dtypes(include=['float64', 'int64']).columns   
            
            print(f"=== Hystplots for the chosen disk: {self.selected_disk_num} ===")
            self.selected_disk_df[numeric_columns].hist(figsize=(15, 10), bins=20, color='skyblue', edgecolor='black')
            
            plt.tight_layout()
            plt.show()
        
        else:   
            features = self.df.select_dtypes(include=['float64', 'int64']).columns if features is None else features 
            n_cols = 4
            n_rows = (len(features) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
            axes = axes.flatten()  
            for idx, feature in enumerate(features):
                sns.histplot(self.df[feature], bins=20, color='skyblue', edgecolor='black', kde=True, ax=axes[idx])
                axes[idx].set_title(f'Hystplot of {feature}')
                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel('Count')
            
            for i in range(len(features), len(axes)):
                axes[i].set_visible(False)
            
            print("=== Hystplot of features for the whole dataset ===")
            plt.tight_layout()
            plt.show()

        
    def plot_boxplots(self, features=None):
        if self.selected_disk_df is None:
            print("You have to select disk before calling. Use select_disk()")
            return
        for i in features: assert i in self.selected_disk_df.columns
        
        print(f"=== Disk chosen: {self.selected_disk_num} ===")
        for i in features: assert i in self.selected_disk_df.columns

        print("=== Boxplots for numeral features for the chosen disk ===")
        numeric_columns = self.selected_disk_df.select_dtypes(include=['float64', 'int64']).columns if features is None else features
        numeric_columns = [i for i in numeric_columns if i != 'capacity_bytes' and i != 'failure'] if features is None else features
        
        plt.figure(figsize=(15, 10))
        self.selected_disk_df[numeric_columns].boxplot()
        plt.xticks(rotation=45)
        plt.title(f'Boxplots for numeric features (selected disk {self.selected_disk_num})')
        plt.show()

    def plot_correlation_matrix(self, features=None):
        if features is not None:
            for i in features: assert i in self.selected_disk_df.columns

        if self.selected_disk_df is not None:
            print(f"=== Correlation matrix for the chosen disk: {self.selected_disk_num} ===")
            numeric_columns = self.selected_disk_df.select_dtypes(include=['float64', 'int64']).columns if features is None else features
            corr_matrix = self.selected_disk_df[numeric_columns].corr()
        else:
            print("=== Correlation matrix for the whole dataset ===")
            numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns if features is None else features
            corr_matrix = self.df[numeric_columns].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
        plt.title(f'Correlation Matrix (selected disk {self.selected_disk_num})' if self.selected_disk_df is not None else 'Correlation Matrix (full dataframe)')
        plt.show()

    def plot_time_series(self, features, multiplot=False):
        if self.selected_disk_df is None:
            print("You have to select disk before calling. Use select_disk()")
            return
        for i in features: assert i in self.selected_disk_df.columns
        
        print(f"=== Chosen disk: {self.selected_disk_num} ===")
        print("=== Time series for the chosen disk ===")
        if not multiplot:
            for feature in features:
                plt.figure(figsize=(10, 6))
                plt.plot(self.selected_disk_df['date'], self.selected_disk_df[feature], label=feature)
                plt.xlabel('Date')
                plt.ylabel(feature)
                plt.xticks(rotation=45)
                plt.title(f'Time series for {feature}')
                plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))  
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) 
                plt.gcf().autofmt_xdate()
                plt.show()
        else:
            plt.figure(figsize=(10, 6))
            for feature in features:
                plt.plot(self.selected_disk_df['date'], self.selected_disk_df[feature], label=feature)
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.xticks(rotation=45)
            plt.title('Time series for the chosen disk')
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))  
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) 
            plt.gcf().autofmt_xdate()
            plt.legend()
            plt.show()
            