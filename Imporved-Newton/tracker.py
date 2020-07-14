from improved import solve
import csv
expression = "np.exp(x)-500"
ans = []
for c in range(500):
    for x in range(500):
        parameters = {'x': x, 'c': c, 'step': 0.00000001}
        response = solve(expression, parameters)
        if response is False:
            continue
        else:
            ans.append([len(response), x, c])

with open("ans.csv", "w") as file:
    writer = csv.writer(file)
    for i in ans:
        writer.writerow(i)
