from improved import solve
import csv
expression = "(x*x*x)-8"


with open("ans1.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(["steps", "dependant vertical", "param horizontal"])
    for c in range(-10, 10, 1):
        for x in range(-50, 50, 1):
            parameters = {'x': x, 'c': c, 'step': 0.00000001}
            response = solve(expression, parameters)
            if response is False:
                continue
            else:
                writer.writerow([len(response), x, c])
