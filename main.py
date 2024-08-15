import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from EDA_class import Data
from pandas import concat
import matplotlib.colors as colors

st.set_page_config(layout="wide")

# Package to read and plot geological models - "welly" - pip install welly

# https://code.agilescientific.com/welly/installation.html

# Usar design pattern para interface gráfica - Model-View-Controller (MVC)
    #    Model: Data // View: Interface // Controller: Interface

plt.rcParams['font.size'] = 8


def read_data(name = 'First', layer:list = None,
              reference_position:float = 0.0):
    # Load the data
    first_shot  = 'Dados/1st_Shoot_temp_baseline_.csv'
    second_shot = 'Dados/2nd_shoot_after_10bbl_of_acid_placement.csv'
    third_shot  = 'Dados/3rd_shoot_traces_after_acid_treatment.csv'
    all_shot    = 'Dados/all_data.csv'

    if   name == 'First': return Data(first_shot, 
                                      layer, 
                                      False, 
                                      reference_position
                                      )
    
    elif name == 'Second': return Data(second_shot, 
                                       layer, 
                                       False, 
                                       reference_position
                                       )
    
    elif name == 'Third': return Data(third_shot, 
                                      layer, 
                                      False, 
                                      reference_position
                                      )
    
    elif name == 'All': return Data(all_shot, 
                                    layer, 
                                    False, 
                                    reference_position
                                    )
    
    else: return Data(name, layer, False, reference_position)


def get_time_labels(data):
    return data.data.columns[1:]


def time_step_index(labels, time_step):
    return labels.index(time_step)


def get_ids(data, init_time_step, end_time_step):
    ids = [time_step_index(data.time_list, init_time_step),
           time_step_index(data.time_list, end_time_step)]
    return [int(i) for i in list(np.arange(ids[0], ids[1]+1))]


def plot_multiple_time_steps_(data, ids):
        print(ids)
        norm = colors.Normalize(vmin = ids[0], 
                                vmax = ids[-1]
                                )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, id in enumerate(ids):
            ax.plot(data.data[data.time_list[id]], 
                    data.data['Depth [m]'],
                    label = data.running_time[id], 
                    linewidth=.5, 
                    color = plt.cm.jet(norm(id))
                    )
            
        ax.set_xlabel('Temperature [°C]'); 
        ax.set_ylabel('Depth [m]')
        fig.gca().invert_yaxis()
        if len(ids) < 10: ax.legend(prop={'size': 8})

        if len(ids) > 10: ax.legend(ncol = 2, 
                                    prop={'size': 8})
            
        if len(ids) > 20: ax.legend(ncol = 3, 
                                    prop={'size': 8})
            
        if len(ids) > 30: ax.legend(ncol = 4, 
                                    prop={'size': 8})
        
        fig.tight_layout()
        return fig


def interface():
    # Title of the app
    st.title('Visualizing Optic Fiber Data')
    st.subheader('This is a simple app to visualize optic fiber data')

    # Add a slider at sidebar to set y-lims:
    st.sidebar.image('.image_fibra-otica.png', width = 450)
    data_name  = st.sidebar.selectbox('Select the data:',
                                      ['First', 'Second', 'Third', 'New'],
                                      index = 0)
    # add a button to read file
    if data_name == 'New':
       data_name = st.sidebar.file_uploader('Upload File',
                                             type = ['.csv', '.rpt']).name

    if data_name != 'All':

        ref_pos = st.sidebar.number_input('Reference Position:')

        data = read_data(data_name, layer = None, reference_position= ref_pos)

        st.sidebar.subheader('SELECT DEPTH RANGE')

        layer_0 = st.sidebar.slider('TOP DEPTH', 
                                    min_value = min(data.depth_list),
                                    max_value = max(data.depth_list) - ref_pos,
                                    value = min(data.depth_list)
                                    )

        layer_0 = st.sidebar.number_input('', 
                                          min_value=min(data.depth_list),
                                          max_value=max(data.depth_list) - ref_pos,
                                          value = layer_0,
                                          step = 10. 
                                          )


        layer_1 = st.sidebar.slider('BOTTOM DEPTH',
                                    min_value=layer_0,
                                    max_value=max(data.depth_list)-ref_pos,
                                    value=max(data.depth_list)-ref_pos
                                    )
        
        layer_1 = st.sidebar.number_input('', 
                                          min_value=layer_0,
                                          max_value=max(data.depth_list)-ref_pos,
                                          value=layer_1, 
                                          step=10.
                                          )

        data = read_data(data_name, layer = [layer_0, layer_1])

        # Add a select box to define the time step:
        st.sidebar.subheader('SELECT TIME STEPS:')

        init_time_step = st.sidebar.selectbox('Select the initial time step:',
                                              data.time_list_id)

        init_time_step = data.time_list[data.time_list_id.index(init_time_step)]

        if st.sidebar.toggle('Plot multiple time steps', False):
            initial_id = time_step_index(data.time_list, init_time_step)
            end_time_step = st.sidebar.selectbox('Select the final time step:',
                                                 data.time_list_id[initial_id:])

            end_time_step = data.time_list[data.time_list_id.index(end_time_step)]

        else: end_time_step = data.time_list[0]

        ids = get_ids(data, init_time_step, end_time_step)

        # Create two columns in the plot view:
        fig_color  = data.plot_contour(); 
        fig_color.set_size_inches(2*4.215, 3)
                ###
        
        if ref_pos != 0:
            labels = [i - ref_pos for i in list(data.data['Depth [m]'])]
            ids_ = list(np.linspace(0, len(labels), 10, 
                                    dtype=int, endpoint=False))
            labels_ = [labels[i] for i in ids_]
            ticks = list(data.data['Depth [m]'].iloc[ids_])
            fig_color.axes[0].set_yticks(ticks, labels=labels_)

        st.pyplot(fig_color, use_container_width=False)
        # Plot selected time steps:
        fig = plot_multiple_time_steps_(data, ids); 
        fig.set_size_inches(2*4, 3);
        fig.axes[0].legend(loc = 'upper right', bbox_to_anchor=(.75, -0.2),
                           ncol = 5, prop={'size': 8})

        ###
        if ref_pos != 0:
            labels = [i - ref_pos for i in list(data.data['Depth [m]'])]
            ids = list(np.linspace(0, len(labels), 10,
                                   dtype=int, endpoint=False))
            labels_ = [labels[i] for i in ids]
            ticks = list(data.data['Depth [m]'].iloc[ids])
            fig.axes[0].set_yticks(ticks, labels=labels_)

        st.pyplot(fig, use_container_width=False)

    

if __name__ == '__main__':
    interface()