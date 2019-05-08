#Limpiar pantalla
from os import system, name
from math import *
import numpy
from pprint import pprint
from matplotlib import pyplot as plt

def clear():
    # Windows
    if name == 'nt':
        _ = system('cls')
    # Max y Linux
    else:
        _ = system('clear')

#Método de bisección
def F_Biseccion():

    def mismosigno(a, b):
        return a * b > 0

    def biseccion(f, a, b):
        assert not mismosigno(f(a), f(b))

        punto_medio = 0
        sig = (a + b) / 2.0
        while(abs(sig - punto_medio) != 0):
            punto_medio = (a + b) / 2.0
            if mismosigno(f(a), f(punto_medio)):
                a = punto_medio
            else:
                b = punto_medio
            sig = (a + b) / 2.0
        return punto_medio

    def f(x):
        return eval(expr)

    expr = input("f(x) = ");
    a = float(input("Punto bajo: "))
    b = float(input("Punto alto: "))

    x = biseccion(f, a, b)
    print("Punto medio:", x)


    input("Presione enter para salir: ")


#Newton-Raphson
def F_Newton_Raphson():

    expr = input("f(x) = ");
    derivada_expr = input("f'(x) = ")
    x0 = float(input("x0 = "))

    def newton_raphson(f, df, xi):
        xsig = xi - f(xi) / df(xi)
        while abs(xsig - xi) != 0:
            xi = xsig
            xsig = xi - f(xi) / df(xi)
        print("Raíz de x:", xsig)

    def f(x):

        return eval(expr)

    def df(x):
        return eval(derivada_expr)

    newton_raphson(f, df, x0)

    input("Presione enter para salir: ")

#Secante
def F_Secante():
    expr = input("f(x) = ");
    xi = float(input("x0 = "))
    ximas1 = float(input("x1 = "))

    def f(x):
        return eval(expr)

    def secante(f, xi, ximas1):

        xsig = ximas1 - ((f(ximas1) * (ximas1-xi)) / (f(ximas1) - f(xi)))
        while abs(xsig - ximas1) != 0:
            xi = ximas1
            ximas1 = xsig
            xsig = ximas1 - ((f(ximas1) * (ximas1-xi)) / (f(ximas1) - f(xi)))
        print("Raíz de x:", xsig)

    secante(f, xi, ximas1)
    input("Presione enter para salir: ")

#Regla falsa
def F_Regla_Falsa():
    expr = input("f(x) = ")
    xI = float(input("x izquierda = "))
    xD = float(input("x derecha = "))

    def f(x):
        return eval(expr)

    def mismoSigno(a, b):
        return a * b > 0

    def reglaFalsa(f, xI, xD):
        assert not mismoSigno(f(xI), f(xD))

        xM = 0
        xMsig = xD - (f(xD) * (xD - xI)) / (f(xD) - f(xI))

        while(abs(xMsig - xM) != 0):
            xM = xMsig
            if mismoSigno(f(xI), f(xM)):
                xI = xM
            else:
                xD = xM
            xMsig = xD - (f(xD) * (xD - xI)) / (f(xD) - f(xI))

        print("Raíz de x: ", xMsig)
    reglaFalsa(f, xI, xD)
    input("Presione enter para salir: ")

#Punto fijo
def F_Punto_Fijo():

    expr = input("g(x) = ")
    x0 = float(input("x0 = "))
    def g(x):
        return eval(expr)

    def puntoFijo(g, x0):
        xsig = g(x0)
        while(abs(xsig - x0) != 0):
            x0 = xsig
            xsig = g(x0)
        print("Raíz de x: ", xsig)

    puntoFijo(g, x0)

    input("Presione enter para salir: ")

#Crout
def F_Crout():
    from pprint import pprint

    def matrixMul(A, B):
        TB = list(zip(*B))
        return [[sum(ea * eb for ea, eb in zip(a, b)) for b in TB] for a in A]

    def pivotize(m):
        """Creates the pivoting matrix for m."""
        n = len(m)
        ID = [[float(i == j) for i in range(n)] for j in range(n)]
        for j in range(n):
            row = max(range(j, n), key=lambda i: abs(m[i][j]))
            if j != row:
                ID[j], ID[row] = ID[row], ID[j]
        return ID

    def lu(A):
        """Decomposes a nxn matrix A by PA=LU and returns L, U and P."""
        n = len(A)
        L = [[0.0] * n for i in range(n)]
        U = [[0.0] * n for i in range(n)]
        P = pivotize(A)
        A2 = matrixMul(P, A)
        for j in range(n):
            L[j][j] = 1.0
            for i in range(j + 1):
                s1 = sum(U[k][j] * L[i][k] for k in range(i))
                U[i][j] = A2[i][j] - s1
            for i in range(j, n):
                s2 = sum(U[k][j] * L[i][k] for k in range(j))
                L[i][j] = (A2[i][j] - s2) / U[j][j]
        return (L, U, P)

    a = [[1, 3, 5], [2, 4, 7], [1, 1, 0]]
    for part in lu(a):
        pprint(part, width=19)
        print()

    input("Presione enter para salir: ")


#Choleski
def F_Choleski():

    def cholesky(A):
        L = [[0.0] * len(A) for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                L[i][j] = sqrt(A[i][i] - s) if (i == j) else \
                    (1.0 / L[j][j] * (A[i][j] - s))

        X = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.transpose(L)), numpy.linalg.inv(L)), b)
        return X



    nn = int(input("Tamaño de filas y columas de la matriz cuadrada: "))

    matrix = []
    b = []
    print("Introduce datos para matriz A por línea: ")

    for i in range(nn):
        a = []
        for j in range(nn):
            a.append(float(input()))
        matrix.append(a)

    print("Introduce datos para el vector B uno por línea: ")
    for i in range(nn):
        b.append(float(input()))

    pprint(cholesky(matrix))

    input("Presione enter para salir: ")

#Método de Euler
def F_Euler():
    expr = input("f'(x,y) = ")
    forTitle = expr
    x0 = float(input("x0: "))
    y0 = float(input("y0: "))
    xf = int(input("Intervalo de x: "))
    n = int(input("Número de puntos: "))
    n = n+1
    deltax = (xf - x0) / (n-1)

    x = numpy.linspace(x0, xf, n)

    y = numpy.zeros([n])
    y[0] = y0
    expr = expr.replace("y", "y[i-1]")
    expr = expr.replace("x", "x[i-1]")
    for i in range(1, n):
        print(expr)
        y[i] = y[i-1] + deltax * eval(expr)

    for i in range(n):
        print(x[i], y[i])

    plt.plot(x, y, 'o')
    plt.xlabel("Valor de x")
    plt.ylabel("Valor de y")
    plt.title(forTitle)
    plt.show()
    input("Presione enter para salir: ")

#Método de Euler mejorado
def F_Euler_Mejorado():
    expr = input("f'(x,y) = ")
    forTitle = expr
    x0 = float(input("x0: "))
    y0 = float(input("y0: "))
    xf = int(input("Intervalo de x: "))
    n = int(input("Número de puntos: "))
    n = n+1
    deltax = (xf - x0) / (n-1)

    x = numpy.linspace(x0, xf, n)
    xM = numpy.linspace(x0, xf, n)

    y = numpy.zeros([n])
    yM = numpy.zeros([n])

    y[0] = y0
    yM[0] = y0

    expr = expr.replace("y", "y[i-1]")
    expr = expr.replace("x", "x[i-1]")
    for i in range(1, n):
        print(expr)
        y[i] = deltax * eval(expr)
        exprM = forTitle
        exprM2 = forTitle

        exprM = exprM.replace("x", "xM[i-1]")
        exprM = exprM.replace("y", "yM[i-1]")

        exprM2 = exprM2.replace("x", "xM[i]")
        exprM2 = exprM2.replace("y", "y[i]")
        print(exprM, exprM2)
        y[i] = y[i - 1] + deltax * eval(expr)
        yM[i] = yM[i-1] + (deltax / 2) * (eval(exprM) + eval(exprM2))

    for i in range(n):
        print(xM[i], yM[i])

    plt.plot(xM, yM, 'o')
    plt.xlabel("Valor de x")
    plt.ylabel("Valor de y")
    plt.title(forTitle)
    plt.show()
    input("Presione enter para salir: ")


#Menú
def menu():
    while(True):
        clear()
        print("1) Bisección")
        print("2) Newton-Raphson")
        print("3) Secante")
        print("4) Regla falsa")
        print("5) Punto fijo")
        print("6) Crout")
        print("7) Choleski")
        print("8) Euler")
        print("9) Euler mejorado")
        print("0) Salir")
        op = input("Opción: ")
        if op == "1":
            F_Biseccion()
        elif op == "2":
            F_Newton_Raphson()
        elif op == "3":
            F_Secante()
        elif op == "4":
            F_Regla_Falsa()
        elif op == "5":
            F_Punto_Fijo()
        elif op == "6":
            F_Crout()
        elif op == "7":
            F_Choleski()
        elif op == "8":
            F_Euler()
        elif op == "9":
            F_Euler_Mejorado()
        elif op == "0":
            print("El programa ha sido cerrado")
            break

menu()



