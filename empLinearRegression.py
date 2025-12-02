import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class empLinearRegression():
    data = None
    w = None
    b = None

    def __init__(self, filename, link=None):
        try:
            self.data = pd.read_csv(filename)
        except FileNotFoundError:
            self.data = pd.read_csv(link)
        self.w = 0
        self.b = 0
    
    def f_x(self, X, W=0, b=0):
        return np.dot(W, X) + b
    
    def J(self, X, y, W, b):
        """
            m: no. of training examples
            J(x, y) = ∑((y_hat - y)^2) / 2m
        """
        m = len(y)
        cost = 0
        for i in range(m):
            cost += ((self.f_x(X[i], W, b) - y[i]) ** 2)
        cost /= (2 * m)
        return cost
    
    def gradient_descent(self, X, y, W = 0, b = 0, alpha=0.0001, iterations=10000):
        m = len(X)
        
        for iteration in range(1, iterations + 1):
            if iteration % 1000 == 0:
                print(f"Iteration: {iteration} | W: {W} | b: {b}")

            dJ_w, dJ_b = 0, 0
            for i in range(m):
                fx = self.f_x(X=X[i], W=W, b=b)
                dJ_w += (fx - y[i]) * X[i]
                dJ_b += (fx - y[i])
            dJ_w, dJ_b = dJ_w / m, dJ_b / m

            W -= alpha * dJ_w
            b -= alpha * dJ_b
        
        return W, b

    def train(self, x, y, alpha=.0001):
        self.w, self.b = self.gradient_descent(x, y, alpha=alpha)

    def visualize(self, x, y, r=False):
        yhat = self.f_x(x, self.w, self.b)
        if not r:
            fig, ax = plt.subplots()
            ax.scatter(x, y)
            ax.plot(x, yhat)
            return fig
        else:
            plt.scatter(x, y)
            plt.plot(x, yhat)
            plt.show()

    def predict(self, x):
        x = [x]
        return round(self.f_x(x, self.w, self.b)[0], 2)



def main():
    model = empLinearRegression(filename="Salary.csv", link="https://raw.githubusercontent.com/Vishal-Singh27/Salary-Predictor/refs/heads/main/Salary.csv")
    x = np.array(model.data["YearsExperience"])
    y = np.array(model.data["Salary"])

    model.train(x, y, alpha=0.01)

    model.visualize(x, y)


if __name__ == "__main__":
    main()