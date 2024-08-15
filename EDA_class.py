import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time

plt.rcParams['axes.grid']   = True
plt.rcParams['grid.alpha']  = 0.3
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size']   = 12

class Data():
    DATA_1_size = 16; DATA_2_size = 10; Data_3_size = 116

    def __init__(self, path, layer:list = None, report = True, reference_position = 0) -> None:
        self.path = path
        self.filetype = self.path[-3:]
        self.reference_position = -1*reference_position
        self.data = self.get_data(layer)

        self.data = self.treat_reference_position()
        self.layer = layer

        self.depth_list = self.data['Depth [m]']
        self.time_list  = list(self.data.columns[1:].values)
        self.running_time = self.util_running_time()
        self.time_list_id = [str(time) + f'({self.running_time[i]})' for i, time in enumerate(self.time_list)]
        if report:
            self.report()
    
    
    def treat_reference_position(self):
        
        if self.reference_position == 0:
            return self.data
        else:
            self.data = self.data[self.data['Depth [m]'] >= self.reference_position]
            self.data['Depth [m]'] = self.data['Depth [m]'] - self.reference_position
            self.data = self.data.reset_index(drop=True)
            
            return self.data

    def get_data(self, layer: list = None):
        """
        Reads and returns data from a file.

        Parameters:
        - layer (list): A tuple containing two values representing the specific nodes to find in the data.

        Returns:
        - data_ (pandas.DataFrame): The data read from the file.

        Raises:
        - ValueError: If the file is not in csv or rpt format.
        """

        if self.path[-3:] == 'csv':
            if layer == None:
                data_ = pd.read_csv(self.path)

            else: 
                data_ = self.find_specific_nodes(pd.read_csv(self.path), layer[0], layer[1])

        elif self.path[-3:] == 'rpt':
            if layer == None:
                data_ = self.read_rpt()
            elif layer != None:
                data  = self.read_rpt()
                data_ = self.find_specific_nodes(data, layer[0], layer[1] )
            else:
                data_ = data

        else:
            raise ValueError('The file must be in csv or rpt format')
        
        return data_
    
    def read_rpt(self):
        """
        Reads and processes an RPT file.

        Returns:
            pandas.DataFrame: A DataFrame containing the processed data from the RPT file.
        """
        with open(self.path, 'r') as file:
            c1 = []
            dados = []
            for i, line in enumerate(file.readlines()[1:]):
                if i == 0:
                    columns = line.split(' ')
                    c1 = [c for c in columns if c != '']
                elif i > 1 and line != '\n':
                    dados.append(line.split('   ')[3:])

            c1[0] = 'Depth [m]'
            c1[-1] = c1[-1][:-1]
            dados = np.array(dados).astype(float)
            df = pd.DataFrame(dados, columns=c1)
            df['Depth [m]'] = abs(df['Depth [m]'])

        return df

    def report(self):
            """
            Prints a report of the data.

            This method prints the path of the data file, the shape of the data, the column names,
            and the first few rows of the data.

            Parameters:
                None

            Returns:
                None
            """
            print(f"Data from: {self.path}")
            print(f"{50 * '#'}")
            print(f"Shape: {self.data.shape}")
            print(f"{50 * '#'}")
            print(f"Columns: \n {self.data.columns}")
            print(f"{50 * '#'}")
            print(f"Head: \n {self.data.head()}")
            print(f"{50 * '#'}")

    def find_specific_nodes(self, data, top_depth, bot_depth) -> pd.DataFrame:
        """
        Find the nodes between the top and bottom depth
        Args:
            data (pd.DataFrame): data to be filtered
            top_depth (int): top depth
            bot_depth (int): bottom depth
        Returns:
            data (pd.DataFrame): filtered data
        """
        return data[(data['Depth [m]'] >= top_depth) & (data['Depth [m]'] <= bot_depth)]

    def util_running_time(self) -> list:
            """
            Calculates the running time based on the data columns
            and a reference time.

            Returns:
                list: A list of running times.
            """
            #  Convert string to time
            columns = self.data.columns[1:]
            if self.filetype == 'csv':
                ref = time.strptime(columns[0], '%m/%d/%Y %H:%M')
                dates = [time.strptime(i, '%m/%d/%Y %H:%M') for i in columns]
                tempo_decorrido = [int(time.mktime(i) - time.mktime(ref)) for i in dates]
            else:
                tempo_decorrido = [float(i) - float(columns[0]) for i in columns]
            return tempo_decorrido
    
    def plot_contour(self, vmin = None, vmax = None, ticks = [] ) -> plt.figure:
        '''
        Plot the contour of the temperature vs depth and time
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)

        vmin = self.data[self.data.columns[1:]].min().min() if vmin is None else vmin
        vmax = self.data[self.data.columns[1:]].max().max() if vmax is None else vmax   

        ticks = np.round(list(np.linspace(vmin, vmax, 10)),2) if ticks is None else ticks

        c = ax.contourf(self.running_time, self.depth_list,
                        self.data.values[:, 1:],
                        vmin = vmin, vmax = vmax,
                        levels=100, cmap='jet')
        
        _ = fig.colorbar(c, ax=ax, label='Temperature [°C]')

        ax.set_xlabel('Time [s]')
        if len(self.running_time) > 20:
            ax.set_xticks(self.running_time[::2], labels=self.running_time[::2], rotation = 90)

        if len(self.running_time) > 40:
            ticks = self.running_time[::6] + [self.running_time[-1]]
            ax.set_xticks(ticks, labels=[int(i) for i in ticks], rotation = 90, fontsize = 8)
            # pass
        else:
            ax.set_xticks(self.running_time, labels=self.running_time, rotation = 90)
        # ax.set_xticklabels(_, rotation = 90)
        ax.set_ylabel('Depth [m]'); ax.grid(False)
        fig.gca().invert_yaxis(); fig.tight_layout()
        return fig

    def plot_all_time_steps(self) -> plt.figure:
        """
        Plot the temperature vs depth for all time steps.

        This method plots the temperature values against the depth values for all time steps in the data.
        Each time step is represented by a different color in the plot.

        Returns:
            matplotlib.figure.Figure: The generated plot figure.
        """
                
        n_time = len(self.time_list)
        norm = colors.Normalize(vmin=0, vmax=n_time)
        fig = plt.subplot()
        for i in range(n_time):
            plt.plot(self.data[self.time_list[i]], self.depth_list,
                     label=self.time_list[i], color=plt.cm.jet(norm(i)))
        plt.xlabel('Temperature [°C]');
        plt.xticks(range(self.time_list[0], self.time_list[1], 20))
        plt.ylabel('Depth [m]')
        plt.gca().invert_yaxis()
        plt.legend(bbox_to_anchor=(1.05, 1))
        return fig

    def plot_specific_time_step(self, time, fig = None, color = None) -> plt.figure:
        """
        Plot the temperature vs depth for a specific time step
        Args:
            time (int): wished time step to plot

        Returns:
            fig: matplotlib figure  
        """
        fig = plt.figure() if fig is None else fig
        ax = fig.add_subplot(111)
        if type(time) == int:       
            label = self.running_time[time]
            ax.plot(self.data[self.time_list[time]],
                    self.depth_list, color = color,
                    label = label)
        else:
            ax.plot(self.data[time], self.depth_list, 
                    color = color,
                    label = f' {self.path} at - {time}')
        ax.set_xlabel('Temperature [°C]')
        ax.set_ylabel('Depth [m]')
        ax.set_xticks(range(self.time_list[0], self.time_list[-1],20))
        fig.gca().invert_yaxis()
        fig.legend(bbox_to_anchor=(1.05, 1))
        return fig
    
    def plot_variation_specific_time_step(self, time: int) -> plt.figure:
        """
        Plot the variation of temperature between two time steps
        Args:
            time (int): wished time step to plot
        
        Returns:
            fig: matplotlib figure
        """
        # Plot variation of temperature between time steps
        fig = plt.subplot()
        plt.plot(self.data[self.time_list[time]]-self.data[self.time_list[time-1]],
                 self.depth_list,
                 label = f' {self.path} at {time+1} - {time}') 
        plt.xlabel('Temperature variation [°C]')
        plt.ylabel('Depth [m]')
        plt.gca().invert_yaxis()
        plt.legend(bbox_to_anchor=(1.05, 1))
        return fig 

    def plot_many_specific_time_steps(self, time_list: list) -> plt.figure:
        """
        Plot the temperature vs depth for many specific time steps
        Args:
            time_list (list): id for wished time steps to plot

        Returns:
            fig: matplotlib figure  
        """
        n_time = len(time_list)
        norm = colors.Normalize(vmin = 0, vmax = n_time)

        fig = plt.subplot(111)
        for i, time in enumerate(time_list):
            fig = self.plot_specific_time_step(time, color = plt.cm.jet(norm(i)))
        return fig
    
    def treat_legend_labels(self, labels):
        """
        Treats the legend labels by splitting them and returning the second part.

        Parameters:
        labels (list): A list of legend labels.

        Returns:
        list: A list of treated legend labels.

        Example:
        >>> obj = EDA_class()
        >>> labels = ['label-1', 'label-2', 'label-3']
        >>> obj.treat_legend_labels(labels)
        ['1', '2', '3']
        """
        return [label.split('-')[1] for label in labels]


if __name__ == '__main__':
    url = 'Dados/3rd_shoot_traces_after_acid_treatment.csv'
    
    data = Data(url, layer = [0, 6000], report=False)

    fig = data.plot_many_specific_time_steps([0,1,2])
    plt.show()