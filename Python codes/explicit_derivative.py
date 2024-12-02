from sympy import symbols, sin, cos, diff

# Define variables
th1, th2, w1, w2 = symbols("th1 th2 w1 w2")
g, m1, m2, l1, l2 = symbols("g m1 m2 l1 l2")

# Define functions
f1 = (-g*(2*m1+m2)*sin(th1)-m2*g*sin(th1-2*th2)-2*sin(th1-th2)*m2*(w2**2*l2+w1**2*l1*cos(th1-th2)))/(l1*(2*m1+m2-m2*cos(2*th1-2*th2)))

f2 = (2*sin(th1-th2)*(w1**2*l1*(m1+m2)+g*(m1+m2)*cos(th1)+w2**2*l2*m2*cos(th1-th2)))/(l2*(2*m1+m2-m2*cos(2*th1-2*th2)))

# Compute Jacobian
jacobian = [
    [diff(f1, var) for var in (th1, th2, w1, w2)],
    [diff(f2, var) for var in (th1, th2, w1, w2)],
]

# Displath2 results
for i, row2 in enumerate(jacobian):
    print(f"Row2 {i + 1}:")
    for j, element in enumerate(row2):
        print(f"  Partial derivative with respect to variable {j + 1}:")
        print(element)
        print("")
