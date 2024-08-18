from tkinter import *
import torch
import torchvision.transforms as transforms
import tkinter as tk
from tkinter import filedialog
from model import CustomViT
from PIL import Image, ImageTk, ImageOps
import timm
global image

def resize_to_canvas(original_width, original_height, canvas_width=360, canvas_height=540):
    aspect_ratio_image = original_width / original_height
    aspect_ratio_canvas = canvas_width / canvas_height

    if aspect_ratio_image > aspect_ratio_canvas:
        new_width = canvas_width
        new_height = canvas_width / aspect_ratio_image
    else:
        new_height = canvas_height
        new_width = canvas_height * aspect_ratio_image

    return int(new_width), int(new_height)

def process_image(img, image_type):
    if image_type == 'Fracture':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5575, 0.5575, 0.5575],
                                 std=[0.2249, 0.2249, 0.2249])
        ])
    elif image_type == 'Original':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3865, 0.3865, 0.3865],
                                 std=[0.2697, 0.2697, 0.2697])
        ])
    return transform(img)

def openAndPut():
    global path
    path = filedialog.askopenfilename(title="Select an Image", filetypes=(("tiff files", ".tif"), ("JPEG files", "*.jpg"),
                                                                               ("PNG files", "*.png"), ("All files", "*.*")))
    global image
    global image_for_mask_multiplication

    if path:
        image = Image.open(path)
        width, height = image.size
        new_width, new_height = resize_to_canvas(width, height)
        image = image.resize((new_width, new_height))
        image_for_mask_multiplication = Image.open(path)
        image = ImageTk.PhotoImage(image)
        original_img.configure(image=image, width=360, height=540)
        original_img.image = image

def open_popup():
    popup = tk.Toplevel(app)
    popup.title("Select Options")
    global image_size_var
    global orientation_var
    global model_type_var
    lb1 = tk.Label(popup, text='Input Image Size')
    lb1.grid(row=0, column=0)
    image_size_var = tk.StringVar(value='Half')
    Whole_size_radio = tk.Radiobutton(popup, text='Original', variable=image_size_var, value='Original')
    Whole_size_radio.grid(row=1, column=0, sticky='w')
    half_size_radio = tk.Radiobutton(popup, text='Half', variable=image_size_var, value='Half')
    half_size_radio.grid(row=1, column=1, sticky='w')

    lb2 = tk.Label(popup, text="Orientation of the image")
    lb2.grid(row=3, column=0, columnspan=2)
    orientation_var = tk.StringVar(value="Left")  # Default is "Left"
    left_radio = tk.Radiobutton(popup, text="Left", variable=orientation_var, value="Left")
    left_radio.grid(row=4, column=0, sticky="w")
    right_radio = tk.Radiobutton(popup, text="Right", variable=orientation_var, value="Right")
    right_radio.grid(row=4, column=1, sticky="w")

    lb3 = tk.Label(popup, text="Model type")
    lb3.grid(row=5, column=0, columnspan = 2)
    model_type_var = tk.StringVar(value='Original')
    Whole_radio = tk.Radiobutton(popup, text='Original', variable=model_type_var, value='Original')
    Whole_radio.grid(row=6, column=0, sticky="w")
    fracture_radio = tk.Radiobutton(popup, text='Fracture', variable=model_type_var, value='Fracture')
    fracture_radio.grid(row=6, column=1, sticky="w")
    Dual_radio = tk.Radiobutton(popup, text='Dual', variable=model_type_var, value='Dual')
    Dual_radio.grid(row=6, column=2, sticky="w")

    # Button to process selections or close popup
    def process_selections():
        # Here you can add your code to process the selections
        print("Image Size:", image_size_var.get())
        print("Orientation:", orientation_var.get())
        print("Model_type: " , model_type_var.get())
        popup.destroy()  # Close the popup after processing

    submit_btn = tk.Button(popup, text="Submit", command=process_selections)
    submit_btn.grid(row=7, column=0, columnspan=2, pady=10)

def open_and_process_file():
    global image, original_image
    original_image = Image.open(path).convert('RGB')

    if image_size_var.get() == 'Original':
        original_img_size = original_image.size
        x_size = original_img_size[0]
        y_size = original_img_size[1]
        half_size = int(x_size / 2)
        if orientation_var.get() == 'Right':
            original_image = original_image.crop((half_size, 0, x_size, y_size))
        elif orientation_var.get() == 'Left':
            original_image = original_image.crop((0, 0, half_size, y_size))
    elif image_size_var.get() == 'Half':
        if orientation_var.get() == "Right":
            original_image = ImageOps.mirror(original_image)
    processed_image = original_image.resize((500, 490), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(processed_image)  # This is now the new PhotoImage object


class fracture_dection:
    def __init__(self):
        popup = Toplevel(app)
        popup.title("Select Site")

        max_width, max_height = 800, 600

        orig_width, orig_height = original_image.size
        self.scaling_factor = min(max_width / orig_width, max_height / orig_height)

        # Resize image for display purposes
        self.resized_image = original_image.resize((int(orig_width * self.scaling_factor), int(orig_height * self.scaling_factor)),
                                              Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(self.resized_image)

        canvas = Canvas(popup, width=self.resized_image.width, height=self.resized_image.height)
        canvas.pack()

        # Display the resized image
        canvas.create_image(0, 0, image=photo, anchor='nw')
        canvas.image = photo  # Keep a reference

        # Initial rectangle coordinates and object
        self.shape = canvas.create_rectangle

        canvas.bind('<ButtonPress-1>', self.onStart)
        canvas.bind('<B1-Motion>', self.onGrow)
        canvas.bind('<ButtonRelease-1>', self.on_release)

    def onStart(self, event):
        self.start = event
        self.drawn = None

    def onGrow(self, event):
        canvas = event.widget
        if self.drawn: canvas.delete(self.drawn)
        self.final = event
        objectId = self.shape(self.start.x, self.start.y, self.final.x, self.final.y, outline='red', width=2)
        self.drawn = objectId

    def on_release(self, event):
        normalized_start_x, normalized_start_y = min(self.start.x, self.final.x), min(self.start.y, self.final.y)
        normalized_end_x, normalized_end_y = max(self.start.x, self.final.x), max(self.start.y, self.final.y)
        self.crop_and_save(normalized_start_x, normalized_start_y, normalized_end_x, normalized_end_y)

    def crop_and_save(self, x1, y1, x2, y2):
        global fracture_image
        # Adjust for scaling factor to crop the original image
        resized_cropped_area = self.resized_image.crop((x1, y1, x2, y2))
        width, height = resized_cropped_area.size
        center_x = 691 + int(355/2)
        center_y = 169 + int(574/2)
        left = center_x - int(width/2)
        top = center_y - int(height / 2)
        image = ImageTk.PhotoImage(resized_cropped_area)

        fracture_img = Label(app, bg='#f6f6bc')
        fracture_img.place(x=left, y=top)
        fracture_img.configure(image=image)
        fracture_img.image = image
        x1, y1, x2, y2 = int(x1 / self.scaling_factor), int(y1 / self.scaling_factor), int(x2 / self.scaling_factor), int(y2 / self.scaling_factor)
        fracture_image = original_image.crop((x1, y1, x2, y2))

def model_processing():
    print(model_type_var.get())
    if model_type_var.get() == 'Original':
        model_path = "./Trained_Model/Original_model.pt"
        threshold = 0.
        sensitivity = 0.309
        specificity = 0.925
    elif model_type_var.get() == 'Fracture':
        model_path = "./Trained_Model/Fracture_model.pt"
        threshold = 0.327566
        sensitivity = 0.687
        specificity = 0.860
    elif model_type_var.get() == 'Dual':
        model_path = "./Trained_Model/Dual_model.pt"
        threshold = 0.199293
        sensitivity = 0.910
        specificity = 0.786

    if model_type_var.get() == 'Dual':  # number of unique locations
        model = CustomViT()
    else:
        model = timm.create_model('vit_base_patch16_224_dino', pretrained=True, num_classes=2)

    state_dict = torch.load(model_path)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    whole_image_processed = process_image(original_image, 'Original')
    original_inputs = whole_image_processed.to(device)
    original_inputs = original_inputs.unsqueeze(0)

    if model_type_var.get() != 'Original':
        fracture_image_processed = process_image(fracture_image, 'Fracture')
        fracture_inputs = fracture_image_processed.to(device)
        fracture_inputs = fracture_inputs.unsqueeze(0)

    if model_type_var.get() == 'Dual':
        outputs = model(original_inputs, fracture_inputs)
    elif model_type_var.get() == 'Original':
        outputs = model(original_inputs)
    else:
        outputs = model(fracture_inputs)
    outputs = torch.sigmoid(outputs)
    pathfxdx_score = outputs[0][1].item()

    if pathfxdx_score > threshold:
        prediction = 'Neoplastic Pathologic Fracture'
    else:
        prediction = 'Non-Pathologic Fracture'

    result_label.config(text=f"PathFxDX_score: {pathfxdx_score:.3f}")
    outcome_label.config(text=f"Prediction: " + prediction)
    specificity_label.config(text = f"Specificity: {specificity:.3f}")
    sensitivity_label.config(text = f"Sensitivity: {sensitivity:.3f}")

app = Tk()
app.title('PathFxDx')
app.geometry("1440x1024")
app.configure(bg="#FFFFFF")

filename = PhotoImage(file = "./GUI_image/PathFxDx_GUI_v4.png")
background_label = Label(app, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

Button(app, text="Select image",font="arial 18 bold", command=openAndPut, bg = '#f2f6f2', highlightthickness = 0, bd = 0).place(x=1067, y = 250, width = 312, height = 50)
Button(app, text="Condition select", font="arial 18 bold", command=open_popup, bg = '#f2f6f2', highlightthickness = 0, bd = 0).place(x=1067, y = 330, width = 312, height = 53)
Button(app, text="Preprocess", font="arial 18 bold", command=open_and_process_file, bg = '#f2f6f2', highlightthickness = 0, bd = 0).place(x=1067, y = 411, width = 312, height = 50)
Button(app, text="Select site", font="arial 18 bold", command=fracture_dection, bg = '#f2f6f2', highlightthickness = 0, bd = 0).place(x=1067, y = 493, width = 312, height = 50)

image = Image.open("./GUI_image/playbutton.png")
width, height = image.size
width_1 = int(width * 0.8)
height_1 = int(height * 0.8)
image = image.resize((width_1, height_1),Image.ANTIALIAS)
file_img = ImageTk.PhotoImage(image)

Button(app, overrelief="solid", command=model_processing,bg="#b5b5b5",image=file_img, highlightbackground = '#b5b5b5', highlightthickness = 0, bd = 0).place(x = 1280, y = 650)

original_img = Label(app, bg = '#95c7bf' )
original_img.place(x=166, y=245)


result_label = Label(app, text="", font = 'arial 18 bold', bg = '#c1bed6')
result_label.place(x = 740, y = 875, width = 600)

outcome_label = Label(app, text="", font = 'arial 20 bold', bg = '#c1bed6')
outcome_label.place(x = 740, y = 825, width = 600)

sensitivity_label = Label(app, text="", font = 'arial 18 ', bg = '#c1bed6')
sensitivity_label.place(x = 710, y = 925, width = 300)

specificity_label = Label(app, text="", font = 'arial 18', bg = '#c1bed6')
specificity_label.place(x = 1010, y = 925, width = 300)

app.mainloop()