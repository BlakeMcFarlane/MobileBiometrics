# Blake McFarlane
# Homework 1 


# Basic operations function which takes in two parameters and prints the required operations.
def basic_operations(a, b):
    print(f'\n- - - - Basic Operations Function - - - -\n')

    # Using an f string to print all 4 operations with new line characters (\n) for styling.
    print(f'Addition: \t\t{a} + {b} = \t{a + b}\nSubtraction: \t\t{a} - {b} = \t{a - b}\nMultiplication: \t{a} x {b} = \t{a * b}\nDivision: \t\t{a} / {b} = \t{a / b:.2f}\n')


# Loop and lists function which prints for 10 positive and even integers.
def loop_and_lists():
    print(f'\n- - - - Loop and Lists Function - - - -\n')

    # Creating a list of integers 1 through 10 using list comprehension.
    numbers = [x for x in range(1,11)]
    print(" Printing even numbers.")
    # Iterating through list and printing all even numbers.
    for x in numbers:
        if x % 2 == 0:
            print(f'  {x}')


# Check number function which asks a user for an integer and tells you whether it is even or odd.
def check_number():
    print(f'\n- - - - Check Number Function - - - -\n')
    integer = assert_integer("Please enter an integer: ", 0)
    
    # Checking sign of integer
    if integer > 0:
        print("- The number is positive.")
    elif integer == 0:
        print("- The number is zero.")
    else:
        print("- The number is negative.")

    # Using modulus operator to only print even numbers
    if (integer % 2 == 0):
        print("- The number is even.")
    else:
        print("- The number is odd.")
    print(f'- - - - - - - - - - - - - - - - - - - - -\n')


# Factorial function which takes an integer parameter and prints the factorial
def factorial(n):
    print(f'\n- - - - - Factorial Function - - - - -\n')
    result = 1

    # Looping through n times
    for i in range(1, n + 1):
        result *= i

    print(f'{n}! = {result}')


# Function to handle user input errors.
def assert_integer(prompt, positive):
    while True:
        # Try, except block which returns false if user does not enter integer.
        try:
            n = int(input(prompt))
            if (n < 0 and positive):
                raise ValueError("Please enter a NON-NEGATIVE number.")
            return n
        except ValueError:
            print("Invalid input. Please enter a valid integer.")


# Main function
def main():
    # Taking user input for basic_operations function
    a = assert_integer("Please enter an integer: ", 0)
    b = assert_integer("Please enter a non-zero number: ", 0)

    # Error handling incase user enters 0
    while b < 1:
        b = assert_integer("Please enter a non-zero number: ", 0)


    # Function call
    basic_operations(a, b)

    # Function call
    loop_and_lists()

    # Function call
    check_number()

    # Taking user input for factorial function

    n = assert_integer("Please enter a non-negative integer: ", 1)

    #function call
    factorial(n)


# Calling main function
main()


    