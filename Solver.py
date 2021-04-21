from sudoku_solver import sudoku_solver

path = input('Enter Image Path : ')
try:
    sudoku_solver(path)
except Exception as e:
    print('Some Error Occurred , '
          'Please Check the specified Path!')