import os
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import ttk, colorchooser, filedialog
import json


class FeatureVisualizationDrawer:
    def __init__(self, root, parameters, visualizations, visualization_image_path):
        self.master = root
        self.parameters = parameters
        self.visualizations = visualizations
        self.visualization_image_path = visualization_image_path
        self.master.title('Feature Visualization')
        self.channel_buttons = []
        self.module_slider = None
        self.parameter_sliders = []
        self.images = []
        self.image_labels = []
        self.current_channel = None

        self.draw_widgets(parameters, visualizations)

        self.module_slider.set(0)
        self.change_module_slider(0)
        self.channel_buttons[0].button.invoke()
        self.master.mainloop()


    def draw_widgets(self, parameters, visualizations):

        # draw title
        title = tk.Frame(self.master, padx=5, pady=5)
        tk.Label(title, text='FeatureVisualization', font=('', 15)).pack()
        title.pack()

        # draw module selection
        module_selection_frame = tk.Frame(self.master, padx = 5, pady = 5)

        channel_button_frame = tk.Frame(module_selection_frame, padx=5, pady=5)
        for _ in parameters["module_to_channel_dict"][parameters["module_list"][0]]:
            button = ChannelButton(self, channel_button_frame)
            self.channel_buttons.append(button)
        channel_button_frame.pack(side=tk.LEFT)
        module_slider_frame = tk.Frame(module_selection_frame, padx=5, pady=5)
        self.module_slider = tk.Scale(
            module_slider_frame,
            from_=0,
            to=len(parameters["module_list"]) - 1,
            showvalue=0,
            command=self.change_module_slider,
            length=300,
            orient=tk.HORIZONTAL)
        self.module_slider.pack()
        module_slider_frame.pack(side=tk.LEFT)
        module_selection_frame.pack()

        # draw parameter selection
        parameter_selection_frame = tk.Frame(self.master, padx=5, pady=5)
        for parameter_name in parameters.keys():
            if parameter_name in ["module_to_channel_dict", "module_list", "epochs"]:
                continue
            parameter_slider = ParameterScale(parameter_selection_frame, self, parameters, parameter_name)
            self.parameter_sliders.append(parameter_slider)
        parameter_selection_frame.pack()

        # draw images
        image_display_frame = tk.Frame(self.master, padx=5, pady=5)
        for epoch in parameters["epochs"]:
            sub_frame = tk.Frame(image_display_frame, padx=5, pady=5)
            path = visualizations[0]["path"]
            path = path.split("/")[-1]
            with Image.open(os.path.join(self.visualization_image_path, path)) as im:
                image = ImageTk.PhotoImage(im)
            self.images.append(image)
            label = tk.Label(sub_frame, image=image)
            label.pack()
            self.image_labels.append(label)
            tk.Label(sub_frame, text=f"epoch {epoch}", font=('', 15)).pack()
            sub_frame.pack(side=tk.LEFT)
        image_display_frame.pack()


    def change_module_slider(self, value):
        self.module_slider.config(label = self.parameters["module_list"][int(value)])
        self.update_channels()


    def update_channels(self):
        module_name = self.parameters["module_list"][self.module_slider.get()]
        for i, channel in enumerate(self.parameters["module_to_channel_dict"][module_name]):
            self.channel_buttons[i].set_channel(channel)


    def redraw_images(self):
        self.images = []
        for epoch, image_label in zip(self.parameters["epochs"], self.image_labels):
            parameter_query_dict = {"epoch" : epoch, "channel" : self.current_channel}
            for parameter_slider in self.parameter_sliders:
                parameter_query_dict[parameter_slider.parameter_name] = parameter_slider.get()
            path = self.get_image_path(parameter_query_dict)
            path = path.split("/")[-1]
            with Image.open(os.path.join(self.visualization_image_path, path)) as im:
                image = ImageTk.PhotoImage(im)
            self.images.append(image)
            image_label.config(image=image)


    def get_image_path(self, parameter_query_dict):
        path = ""
        for parameter_dict in self.visualizations:
            mismatches = False
            for key, value in parameter_query_dict.items():
                if float(parameter_dict[key]) != float(value):
                    mismatches = True
                    break
            if not mismatches:
                path = parameter_dict["path"]
                break
        return path


class ChannelButton:
    def __init__(self, feature_visualization_drawer, master):
        self.frame = tk.Frame(master, padx=5, pady=5)
        self.feature_visualization_drawer = feature_visualization_drawer
        self.button = tk.Button(self.frame, command=self.on_press)
        self.button.pack()
        self.label = tk.Label(self.frame, font=('', 12))
        self.label.pack()
        self.channel = None
        self.image = None
        self.frame.pack(side=tk.LEFT)


    def set_channel(self, channel):
        self.channel = channel
        self.label.config(text=f"{channel}")
        parameter_query_dict = {"epoch": self.feature_visualization_drawer.parameters["epochs"][-1], "channel": channel}
        for parameter_slider in self.feature_visualization_drawer.parameter_sliders:
            parameter_query_dict[parameter_slider.parameter_name] = parameter_slider.get()
        path = self.feature_visualization_drawer.get_image_path(parameter_query_dict)
        path = path.split("/")[-1]
        with Image.open(os.path.join(self.visualization_image_path, path)) as im:
            self.image = ImageTk.PhotoImage(im)
        self.button.config(image=self.image)


    def on_press(self):
        self.feature_visualization_drawer.current_channel = self.channel
        self.feature_visualization_drawer.redraw_images()


class ParameterScale:
    def __init__(self, master, feature_visualization_drawer, parameters, parameter_name):
        self.feature_visualization_drawer = feature_visualization_drawer
        self.parameter_name = parameter_name
        self.parameter_list = parameters[parameter_name]
        self.feature_visualization_drawer = feature_visualization_drawer

        frame = tk.Frame(master, padx=5, pady=5)

        self.scale = tk.Scale(
            frame,
            from_=0,
            to=len(self.parameter_list) - 1,
            command=self.parameter_changed,
            showvalue=0,
            label=parameter_name,
            orient=tk.HORIZONTAL)
        self.scale.pack()
        self.scale.set(0)

        self.label = tk.Label(frame, font=('', 12))
        self.label.pack()

        frame.pack(side=tk.LEFT)


    def parameter_changed(self, value):
        self.label.config(text=f"{self.parameter_list[int(value)]}")
        self.feature_visualization_drawer.redraw_images()
        self.feature_visualization_drawer.update_channels()


    def get(self):
        return self.parameter_list[int(self.scale.get())]


def start_app(visualization_image_path):
    with open(os.path.join(visualization_image_path, "meta.json"), "r") as outfile:
        json_object = json.load(outfile)
    
    my_parameters = json_object['parameters']
    visualizations = json_object['visualizations']

    my_root = tk.Tk()
    feature_visualization_drawer = FeatureVisualizationDrawer(my_root, my_parameters, visualizations)
    return feature_visualization_drawer
