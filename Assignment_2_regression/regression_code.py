import torch
from torch import nn
import numpy as np
import math
import matplotlib.pyplot as plt
import code
# code.interact(local=locals())
from sklearn.model_selection import KFold

def poly_train_mse_loss(y_init, y_hat):
    loss_fn = nn.MSELoss()
    train_err = loss_fn(y_init, y_hat)
    # print("Train error (MSE loss) :", train_err.item())
    return train_err.item()

def poly_predict_y_hat(x_values, w_lin):
    y_hat = torch.matmul(x_values, w_lin)
    return y_hat

def poly_train_w(x_values, y_values):
    x_pinv = torch.linalg.pinv(x_values)
    w_lin = torch.matmul(x_pinv, y_values)
    # print("w_lin:", w_lin)
    return w_lin

def poly_kfold_cross_validation(current_degree, n_splits, x_values, y_values, loss_fn):
    # poly KFOLD cross validation
    kfold = KFold(n_splits=n_splits)
    kfold_loss_list = []
    xy_values = torch.cat((x_values, y_values), dim=1)
    for fold_i, (train_ids, val_ids) in enumerate(kfold.split(xy_values)):
        
        # print("Fold Info:")
        # print(fold_i, (train_ids, val_ids))

        x_values_train = xy_values[:, :current_degree+1][train_ids]
        y_values_train = xy_values[:, current_degree+1][train_ids].unsqueeze(1)

        x_values_val = xy_values[:, :current_degree+1][val_ids]
        y_values_val = xy_values[:, current_degree+1][val_ids].unsqueeze(1)

        # w
        w_lin = poly_train_w(x_values_train, y_values_train)

        # predict
        y_hat = poly_predict_y_hat(x_values_val, w_lin)

        # loss
        kfold_loss = loss_fn(y_values_val, y_hat)
        kfold_loss_list.append(kfold_loss)
    
    return sum(kfold_loss_list) / len(kfold_loss_list)

def poly_regression_plot(current_degree, x_init, y_init, y_hat):
    # Plot dots
    plt.plot(x_init, y_init, 'o', label='Data')

    # Plot line
    plt.plot(x_init, y_hat, '-', label='Prediction')

    # Add legend
    plt.legend()

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Regression: Degree ' + str(current_degree))

    # Display the plot
    plt.show()

def poly_regression(current_degree, x_init, y_init, n_splits, loss_fn):

    x_values = torch.ones(n_points, current_degree + 1)
    for i in range(current_degree):
        x_values[:, i] = x_init ** (current_degree - i)
    y_values = y_init.unsqueeze(1)

    x_pinv = torch.linalg.pinv(x_values)
    w_lin = torch.matmul(x_pinv, y_values)
    # print("w_lin:", w_lin)

    y_hat = poly_predict_y_hat(x_values, w_lin)

    poly_regression_plot(current_degree, x_init, y_init, y_hat)

    train_mse_loss = loss_fn(y_values, y_hat)
    # print("--> MSE Loss : " + str(train_err))

    kfold_avg_loss = poly_kfold_cross_validation(
        current_degree=current_degree,
        n_splits=n_splits,
        x_values=x_values,
        y_values=y_values,
        loss_fn=loss_fn
    )
    # print("--> Kfold average MSE Loss : " + str(kfold_avg_loss))
    return train_mse_loss, kfold_avg_loss

def poly_regu(degree, lambd, x_values, y_values):
    A = torch.matmul(x_values.transpose(0, 1), x_values)
    B = torch.inverse(A - lambd * torch.eye(degree+1))
    w_reg = torch.matmul(torch.matmul(B, x_values.transpose(0, 1)), y_values)
    return w_reg

# y = 2 * x + epsilon

n_points = 15
x_min, x_max = -3, 3
x_init = torch.linspace(x_min, x_max, n_points)
epsilon = torch.randn(n_points)
y_init = 2 * x_init + epsilon

print("x_init:", x_init)
print("y_init:", y_init)


degree = [1, 5, 10, 14]
n_splits_list = [5, 5, 5, 5]

for d in range(len(degree)):
    print("DEGREE --> " + str(degree[d]))
    train_mse_loss, kfold_avg_loss = poly_regression(
        degree[d],
        x_init,
        y_init,
        n_splits_list[d],
        poly_train_mse_loss
    )
    print("train_mse_loss:", train_mse_loss)
    print("kfold_avg_loss:", kfold_avg_loss)

# y = sin(2*pi) + epsilon

n_points = 15
x_min, x_max = 0, 1
x_init = torch.linspace(x_min, x_max, n_points)
epsilon = torch.randn(n_points) * math.sqrt(0.04)
y_init = torch.sin(2 * math.pi * x_init) + epsilon

print("x_init:", x_init)
print("y_init:", y_init)

degree = [1, 5, 10, 14]
n_splits_list = [5, 5, 5, 5]

for d in range(len(degree)):
    print("DEGREE --> " + str(degree[d]))
    train_mse_loss, kfold_avg_loss = poly_regression(
        degree[d],
        x_init,
        y_init,
        n_splits_list[d],
        poly_train_mse_loss
    )
    print("train_mse_loss:", train_mse_loss)
    print("kfold_avg_loss:", kfold_avg_loss)

# varying y = sin(2*pi) + epsilon 's data point
# m(n_points) = 10, 15, 80, 320

degree = [14]
n_splits_list = [5]
n_points_list = [10, 15, 80, 320]
for i in range(len(n_points_list)):
    
    print("Total data point = " + str(n_points_list[i]))

    n_points = n_points_list[i]
    x_min, x_max = 0, 1
    x_init = torch.linspace(x_min, x_max, n_points)
    epsilon = torch.randn(n_points) * math.sqrt(0.04)
    y_init = torch.sin(2 * math.pi * x_init) + epsilon

    # print("x_init:", x_init)
    # print("y_init:", y_init)

    for d in range(len(degree)):
        print("DEGREE --> " + str(degree[d]))
        train_mse_loss, kfold_avg_loss = poly_regression(
            degree[d],
            x_init,
            y_init,
            n_splits_list[d],
            poly_train_mse_loss
        )
        print("train_mse_loss:", train_mse_loss)
        print("kfold_avg_loss:", kfold_avg_loss)

# regu

n_points = 15
x_min, x_max = 0, 1
x_init = torch.linspace(x_min, x_max, n_points)
epsilon = torch.randn(n_points) * math.sqrt(0.04)
y_init = torch.sin(2 * math.pi * x_init) + epsilon

degree = [14]
n_splits_list = [5]
lambda_list = [0, 0.001 / n_points, 1 / n_points, 1000 / n_points]

for l in range(len(lambda_list)):

    print("Lambda:", lambda_list[l])

    for d in range(len(degree)):
        
        print("DEGREE --> " + str(degree[d]))

        current_degree = degree[d]
        
        x_values = torch.ones(n_points, current_degree + 1)
        for i in range(current_degree):
            x_values[:, i] = x_init ** (current_degree - i)
        y_values = y_init.unsqueeze(1)

        w_reg = poly_regu(current_degree, lambda_list[l], x_values, y_values)
        
        # x_pinv = torch.linalg.pinv(x_values)
        # w_lin = torch.matmul(x_pinv, y_values)
        # print("w_lin:", w_lin)

        y_hat = poly_predict_y_hat(x_values, w_reg)

        poly_regression_plot(current_degree, x_init, y_init, y_hat)

        train_mse_loss = poly_train_mse_loss(y_values, y_hat)
        print("train_mse_loss:", train_mse_loss)

        # poly KFOLD cross validation
        kfold = KFold(n_splits=n_splits_list[d])
        kfold_loss_list = []
        xy_values = torch.cat((x_values, y_values), dim=1)
        for fold_i, (train_ids, val_ids) in enumerate(kfold.split(xy_values)):
            
            # print("Fold Info:")
            # print(fold_i, (train_ids, val_ids))

            x_values_train = xy_values[:, :current_degree+1][train_ids]
            y_values_train = xy_values[:, current_degree+1][train_ids].unsqueeze(1)

            x_values_val = xy_values[:, :current_degree+1][val_ids]
            y_values_val = xy_values[:, current_degree+1][val_ids].unsqueeze(1)

            # w
            w_reg = poly_regu(current_degree, lambda_list[l], x_values_train, y_values_train)

            # predict
            y_hat = poly_predict_y_hat(x_values_val, w_reg)

            # loss
            kfold_loss = poly_train_mse_loss(y_values_val, y_hat)
            kfold_loss_list.append(kfold_loss)
        
        print("--> Kfold average MSE Loss : " + str(sum(kfold_loss_list) / len(kfold_loss_list)))

