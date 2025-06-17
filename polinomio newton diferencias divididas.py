!pip install -q ipywidgets

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets
from IPython.display import display

def diferencias_divididas(x, y):
    n = len(x)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
    return coef
def evaluar_newton(x_data, coef, x_interp):
    n = len(x_data)
    p = coef[0]
    for k in range(1, n):
        p += coef[k] * np.prod(x_interp - x_data[:k])
    return p
def interpolacion_newton(x_str, y_str, x_val):
    try:
        x_vals = list(map(float, x_str.split(',')))
        y_vals = list(map(float, y_str.split(',')))
        x_val_interp = float(x_val)
    except:
        print(" Asegúrate de ingresar números separados por comas.")
        return

    if len(x_vals) != len(y_vals):
        print(" Las listas de x y f(x) deben tener la misma longitud.")
        return

    coef = diferencias_divididas(np.array(x_vals), np.array(y_vals))
    resultado = evaluar_newton(np.array(x_vals), coef, x_val_interp)

  
    print(f"\n Valor interpolado en x = {x_val_interp}: {resultado:.5f}")

  
    x_plot = np.linspace(min(x_vals), max(x_vals), 200)
    y_plot = [evaluar_newton(np.array(x_vals), coef, xp) for xp in x_plot]

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, y_plot, label='Polinomio interpolante', color='blue')
    plt.scatter(x_vals, y_vals, color='red', label='Puntos dados')
    plt.axvline(x_val_interp, color='green', linestyle='--', label=f'x = {x_val_interp}')
    plt.axhline(resultado, color='gray', linestyle='--')
    plt.title('Interpolación por Polinomio de Newton')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

x_input = widgets.Text(value='1,2,4', description='x:')
y_input = widgets.Text(value='1,4,16', description='f(x):')
val_input = widgets.Text(value='3', description='x_interp:')
interact(interpolacion_newton, x_str=x_input, y_str=y_input, x_val=val_input)