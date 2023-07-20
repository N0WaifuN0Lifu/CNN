#This piece of code generates a random polynomial between order 0 and 10, plots it, and saves the plot as a .png file in "images" folder with the name being the digit in front of each polynomial term, with the highest order term first.

import numpy as np
import matplotlib.pyplot as plt
import random
import os

#This function generates a random polynomial between order 0 and 9
def generate_polynomial():
    order = random.randint(0,3)
    coefficients = np.random.randint(0,9,order+1)
    polynomial = np.poly1d(coefficients)
    return polynomial

#This function plots the polynomial and saves the plot as a .png file in "images" folder with the name of the image being every constant in the polynomial expression
def plot_polynomial(polynomial):
    x = np.linspace(-10,10,100)
    y = polynomial(x)
    plt.plot(x,y)
    plt.grid()
    #turns off axis
    plt.axis('off')
    #turn off the grid lines
    plt.grid(False)
    #change color to black, thicken line
    plt.plot(x,y, color='black', linewidth=4)
    #saves the fig, limiting the dimensions to 128*128
    plt.savefig("images/" + convert_to_list(str(polynomial).replace(" ","").replace("x","").replace("^","").replace("\n","")) + ".png" , bbox_inches='tight', pad_inches=0,dpi=64)
    plt.close()

#converts a number in this form 765328+9+4+7+8+1+7 to [76532][8947817]
def convert_to_list(number):
    number = str(number)
    if len(number) == 1:
        return("[0]" + "[" + str(number) + "]")
    for i in range(len(number)):
        if number[i] == '+':
            lst1 = number[:i-1]
            lst1 = lst1 + "1"
            lst2 = number[i-1:]
            return("[" + (lst1) + "]" + "[" + (lst2).replace("+","") + "]")
#This function creates a folder named "images" if it does not exist
def create_folder():
    if not os.path.exists("images"):
        os.makedirs("images")


#This function iterates through every image in images, and creates appends the second digit in them to a CSV file
def create_csv():
    i = 0
    k = 0
    for filename in os.listdir("images"):
        i += 1
        if filename.endswith(".png"):
            with open("order.csv", "a") as file:
                file.write(filename[1] + "\n")
                k += 1
            continue
        else:
            continue
    print(i)
    print(k)


#generate n plots
def generate_n_plots(n):
    k = 0
    while k < n:
        print(f"currently making graph number {k} out of {n}")
        try:
            plot_polynomial(generate_polynomial())
            k += 1
        except:
            print("error")
            continue
#delete all images in images folder
def delete_images():
    for filename in os.listdir("images"):
        if filename.endswith(".png"):
            os.remove("images/" + filename)
            continue

#delete csv titled order.csv
def delete_csv():
    os.remove("order.csv")


create_folder()
delete_images()
delete_csv()
generate_n_plots(1000)
create_csv()