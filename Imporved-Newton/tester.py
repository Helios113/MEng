from scipy import optimize
import csv
import os
import sqlite3
from sqlite3 import Error



l = """function,x0, x1,
  newton_iterations  ,
  newton_cals  ,
  exnewton_iterations  ,
  exnewton_cals  ,
  secant_iterations  ,
  secant_cals  ,
  halley_iterations  ,
  halley_cals  """


create_data_table = """
CREATE TABLE IF NOT EXISTS data (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  function TEXT NOT NULL,
  x0 NUMERIC,
  x1 NUMERIC,
  newton_iterations INTEGER,
  newton_cals INTEGER,
  exnewton_iterations INTEGER,
  exnewton_cals INTEGER,
  secant_iterations INTEGER,
  secant_cals INTEGER,
  halley_iterations INTEGER,
  halley_cals INTEGER
);
"""





"""
Variables to be used
"""
dataPath = r'results'
fileName = "/info.db"

"""
function and its first and second derivative
"""
def f(x):
    return (x**3 - 1)  # only one real root at x = 1
def fprime(x):
    return (3*x**2)
def fprime2(x):
    return (6*x)



"""
SQL functions
"""

def create_connection(path):
    connection = None
    try:
        connection = sqlite3.connect(path)
        print("Connection to SQLite DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully")
        return cursor.fetchall()
    except Error as e:
        print(f"The error '{e}' occurred")



"""
TEST function
"""
def test(X0, X1, func, fList):

    #bracket = TEST_BRACKET
    x0 = X0

    x1 = X1

    ans = [func, X0,X1]
    #bisect
    #ans+=setData(optimize.root_scalar(fList[0],bracket = bracket, method='bisect'))
    #brentq
    #ans+=setData(optimize.root_scalar(fList[0],bracket = bracket, method='brentq'))
    #brenth
    #ans+=setData(optimize.root_scalar(fList[0],bracket = bracket, method='brenth'))
    #ridder
    #ans+=setData(optimize.root_scalar(fList[0],bracket = bracket, method='ridder'))
    #toms748
    #ans+=setData(optimize.root_scalar(fList[0],bracket = bracket, method='toms748'))
    #newton
    try:
        ans+=setData(optimize.root_scalar(fList[0], x0=x0, x1 = x1, fprime=fList[1], method='newton'))
    except:
        ans+=[-1,-1]

    #exnewton
    try:
        ans+=setData(optimize.root_scalar(fList[0], x0=x0, x1 = x1, fprime=fList[1], method='exnewton'))
    except:
        ans+=[-1,-1]

    #secant
    try:
        ans+=setData(optimize.root_scalar(fList[0], x0=x0, x1 = x1, fprime=fList[1], method='secant'))
    except:
        ans+=[-1,-1]

    #halley
    try:
        ans+=setData(optimize.root_scalar(fList[0], x0=x0, x1 = x1, fprime=fList[1], fprime2 = fList[2], method='halley'))
    except:
        ans+=[-1,-1]

    return ans

def setData(a):
    ans = []
    if a.converged:
        ans.append(a.iterations)
        ans.append(a.function_calls)
    else:
        ans.append(-1)
        ans.append(-1)
    return ans

def writeToDB(dataPath, fileName, data):
    connection = create_connection(dataPath+fileName)
    global l
    global create_data_table
    execute_query(connection, create_data_table)
    for i in data:
        i[0] = "\'"+i[0]+"\'"
        #i[1] = "\'"+str(i[1])+"\'"
        func, x0, x1 = [i[j] for j in range(3)] 
        msg = f"SELECT * FROM data WHERE function={func} AND x0 = {x0} AND x1 = {x1}"
        if execute_query(connection, msg) == []:
            s = ","
            s = s.join([str(elem) for elem in i])
            print(s)
            msg = f"INSERT INTO data ({l}) VALUES({s})"
            print(msg)
            execute_query(connection, msg)



fList = [f, fprime, fprime2]
testData = [test(-600,501, "x^3-1",fList)]

print(testData)
writeToDB(dataPath,fileName,testData)





connection = create_connection(dataPath+fileName)


"""
fList = [f, fprime, fprime2]
testData = [test([0,2],0.2,2, "x^3-1",fList)]

writeToCSV(dataPath,fileName, testData)
"""


