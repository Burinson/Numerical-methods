#Limpiar pantalla
from os import system, name
from math import *
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
    a = float(input("Punto bajo:"))
    b = float(input("Punto alto:"))

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

#Menú
def menu():
    while(True):
        clear()
        print("1) Bisección")
        print("2) Newton-Raphson")
        print("3) Secante")
        print("4) Regla falsa")
        print("5) Punto fijo")
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
        elif op == "0":
            print("El programa ha sido cerrado")
            break

menu()



