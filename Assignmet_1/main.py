import math

def determinant(m: list[list[float]]) -> float:
    return m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0])

def trace(m: list[list[float]]) -> float:
    return m[0][0] + m[1][1] + m[2][2]

def norm(v: list[float]) -> float:
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) 

def transpose(m: list[list[float]]) -> list[list[float]]:
    return [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]

def multiply(m: list[list[float]], v: list[float]) -> list[float]:
    return [m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2], m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2], m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2]]

def solveCramer(m: list[list[float]], v: list[float]):
    mX = [[v[0], m[0][1], m[0][2]], [v[1], m[1][1], m[1][2]], [v[2], m[2][1], m[2][2]]]
    mY = [[m[0][0], v[0], m[0][2]], [m[1][0], v[1], m[1][2]], [m[2][0], v[2], m[2][2]]]
    mZ = [[m[0][0], m[0][1], v[0]], [m[1][0], m[1][1], v[1]], [m[2][0], m[2][1], v[2]]]
    det = determinant(m)

    x = float(determinant(mX) / det)
    y = float(determinant(mY) / det)
    z = float(determinant(mZ) / det)

    print("Cramer Solution:")
    print("x = {}\ny = {}\nz = {}\n".format(round(x,2), round(y,2), round(x,2)))

def solveInversion(m: list[list[float]], v: list[float]):
    mX = [[m[0][0], m[0][1], m[0][2]], 
          [m[1][0], m[1][1], m[1][2]], 
          [m[2][0], m[2][1], m[2][2]]]
    cofactorMatrix = [
        [m[1][1]*m[2][2]-m[1][2]*m[2][1], m[1][2]*m[2][0]-m[1][0]*m[2][2], m[1][0]*m[2][1]-m[1][1]*m[2][0]],
        [m[0][2]*m[2][1]-m[0][1]*m[2][2], m[0][0]*m[2][2]-m[0][2]*m[2][0], m[0][1]*m[2][0]-m[0][0]*m[2][1]],
        [m[0][1]*m[1][2]-m[0][2]*m[1][1], m[0][2]*m[1][0]-m[0][0]*m[1][2], m[0][0]*m[1][1]-m[0][1]*m[1][0]]
    ]

    adjM = transpose(cofactorMatrix)
    det = float(1 / determinant(m))

    inversionMatrix = [
        [adjM[0][0] * det, adjM[0][1] * det, adjM[0][2] * det],
        [adjM[1][0] * det, adjM[1][1] * det, adjM[1][2] * det],
        [adjM[2][0] * det, adjM[2][1] * det, adjM[2][2] * det]
    ]

    solution = multiply(inversionMatrix, v)
    print("Inversion Solution:")
    print("x = {}\ny = {}\nz = {}\n".format(round(solution[0],2), round(solution[1],2), round(solution[2],2)))


A = []
B = []
with open("input.txt", "r") as input:
    for line in input:
        B.append(float(line[line.find("=")+1:]))
        auxList = []
        variables = ["x", "y", "z"]
        start = {"x": 0, "y": line.find("x")+1, "z": line.find("y")+1}
        end = {"x": line.find("x"), "y": line.find("y"), "z": line.find("z")}
        for var in variables:
            if line.find(var) != -1:
                aux1 = line[start[var]:end[var]]
                aux2 = aux1.replace(" ","")
                num = aux2.replace("+","")
                if num == "":
                    num = 1
                elif num == "-":
                    num = -1
                num = float(num)
                auxList.append(num)
            else:
                auxList.append(0)
        A.append(auxList)

solveCramer(A, B)
solveInversion(A, B)
