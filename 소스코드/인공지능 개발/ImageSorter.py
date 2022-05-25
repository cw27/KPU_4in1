from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import shutil
import os

# This will forward the pointer to the next image
def forward(image_number):
   global my_label
   global button_forward
   global button_back
   global button_upload
   global status
   global button_delete
   # recreating the labels and buttons
   if image_number < len(my_img) - 1:
         my_label.grid_forget()
         image_number += 1
         my_label = Label(image=my_img[image_number])
         my_label.grid(row=0, column=0, columnspan=3)
         button_forward = Button(root, text="Next >>", command=lambda: forward(image_number))
         button_back = Button(root, text="<< Prev", command=lambda: back(image_number))
         status = Label(root, text="Image " + str(image_number+1) + " of " + str(len(my_img)), bd=1, relief=SUNKEN, anchor=E)
         status.grid(row=3, column=0, columnspan=3, sticky=W+E)
         button_forward.grid(row=1, column=2)
         button_back.grid(row=1, column=0)
         button_upload = Button(root, text="Upload Image", command=lambda: upload_image(my_img))
         button_upload.grid(row=1, column=1)
         button_delete = Button(root, text="Delete Image",
                                command=lambda: delete_image(my_img, my_img_files, image_number))
         button_delete.grid(row=2, column=1)
   else:
       button_forward = Button(root, text="Next >>", state=DISABLED)
       button_forward.grid(row=1, column=2)

# This function will move the pointer back to the previous image in the list
def back(image_number):
    global my_label
    global button_forward
    global button_back
    global button_upload
    global status
    global button_delete
    # Recreating labels and buttons
    if image_number > 0:
        my_label.grid_forget()
        image_number -= 1
        my_label = Label(image=my_img[image_number])
        my_label.grid(row=0, column=0, columnspan=3)
        button_forward = Button(root, text="Next >>", command=lambda: forward(image_number))
        button_back = Button(root, text="<< Prev", command=lambda: back(image_number))
        status = Label(root, text="Image " + str(image_number + 1) + " of " + str(len(my_img)), bd=1, relief=SUNKEN, anchor=E)
        status.grid(row=3, column=0, columnspan=3, sticky=W+E)
        button_forward.grid(row=1, column=2)
        button_back.grid(row=1, column=0)
        button_upload = Button(root, text="Upload Image", command=lambda: upload_image(my_img))
        button_upload.grid(row=1, column=1)
        button_delete = Button(root, text="Delete Image", command=lambda: delete_image(my_img, my_img_files, image_number))
        button_delete.grid(row=2, column=1)
    else:
        button_back = Button(root, text="<< Prev", state=DISABLED)
        button_back.grid(row=1, column=0)


#Uploading image
def upload_image(my_img):
    root.filename = filedialog.askopenfilename(initialdir="../app", title="Select a file",
                                               filetypes=(("JPEG files", "*.jpg"),
                                                          ("PNG files", "*.png")))
    if os.path.isfile(root.filename):
        shutil.copy(root.filename, "./images/1/")
        head, tail = os.path.split(root.filename)
        new_name = "./images/1/"+tail
        my_img_files.append(Image.open(new_name))
        my_img.append([ImageTk.PhotoImage(Image.open(new_name))])


#Loading images from the directory
def load_images(my_img, my_img_files):
    directory = "./images/1/"
    counter = 0
    for file in os.listdir(directory):
        if file.endswith(".jpg") or file.endswith(".png"):
            my_img_files.append(Image.open("./images/1/"+file))
            my_img.append(ImageTk.PhotoImage(my_img_files[counter]))
            counter += 1

#This function will delete the image from both the array and the directory
def delete_image(my_img, my_img_files, index):
    #for file in my_img_files:
        #print(file.filename)
    os.remove(my_img_files[index].filename)
    del(my_img_files[index])
    del(my_img[index])
    if len(my_img) - index > 0:
        forward(index)
    else:
        back(index)

def Exit(event):
    root.destroy()

root = Tk()
root.title('Images Sorter')
root.geometry("500x500")
root.resizable(0, 0)

#Our static list of images
my_img = []
my_img_files = []
load_images(my_img, my_img_files)
status = Label(root, text="Image 1 of " + str(len(my_img)), bd=1, relief=SUNKEN, anchor=E)
my_label = Label(image=my_img[0])
my_label.grid(row=0, column=0, columnspan=3)



button_back = Button(root, text="<< Prev", command=back, state=DISABLED)
button_delete = Button(root, text="Delete Image", command=lambda: delete_image(my_img, my_img_files, 0))
button_forward = Button(root, text="Next >>", command=lambda: forward(0))
button_upload = Button(root, text="Upload Image", command=lambda: upload_image(my_img))

button_back.grid(row=1, column=0)
button_delete.grid(row=2, column=1)
button_forward.grid(row=1, column=2, pady=10)
button_upload.grid(row=1, column=1)
status.grid(row=3, column=0, columnspan=3, sticky=W+E)

my_label.focus_set()
root.bind("<Left>", lambda event:back)
root.bind("<Right>", lambda event:forward)
root.bind("<Delete>", lambda event:delete_image)
root.bind("<Escape>", lambda event:Exit)

root.mainloop()