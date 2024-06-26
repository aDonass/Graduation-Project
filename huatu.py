import matplotlib.pyplot as plt

# Your new results
steps = list(range(10000, 151000, 1000))
train_losses = [
    2.65e-01, 2.13e-01, 1.51e-01, 1.14e-01, 1.13e-01, 8.95e-02, 7.57e-02, 6.44e-02, 5.72e-02, 6.40e-02,
    5.97e-02, 6.30e-02, 6.73e-02, 4.87e-02, 5.34e-02, 4.03e-02, 4.30e-02, 4.46e-02, 3.91e-02, 3.73e-02,
    3.78e-02, 3.56e-02, 4.33e-02, 4.58e-02, 3.22e-02, 3.36e-02, 3.27e-02, 3.36e-02, 2.95e-02, 2.87e-02,
    2.96e-02, 2.28e-02, 2.34e-02, 2.81e-02, 2.59e-02, 2.71e-02, 2.11e-02, 3.00e-02, 2.08e-02, 1.77e-02,
    2.89e-02, 1.96e-02, 1.70e-02, 1.83e-02, 1.67e-02, 1.84e-02, 1.88e-02, 2.27e-02, 1.49e-02, 1.43e-02,
    1.47e-02, 1.49e-02, 1.37e-02, 1.98e-02, 1.39e-02, 1.71e-02, 1.56e-02, 1.31e-02, 1.38e-02, 1.29e-02,
    1.27e-02, 1.51e-02, 1.03e-02, 1.16e-02, 1.18e-02, 1.27e-02, 1.05e-02, 1.11e-02, 1.24e-02, 9.99e-03,
    1.02e-02, 9.50e-03, 1.06e-02, 9.65e-03, 9.37e-03, 1.36e-02, 8.29e-03, 8.76e-03, 9.12e-03, 9.42e-03,
    9.09e-03, 8.04e-03, 7.71e-03, 9.43e-03, 7.71e-03, 1.03e-02, 8.62e-03, 9.45e-03, 8.89e-03, 9.45e-03,
    8.03e-03, 7.65e-03, 8.68e-03, 6.87e-03, 7.27e-03, 7.63e-03, 7.40e-03, 7.03e-03, 7.08e-03, 7.27e-03,
    8.96e-03, 6.74e-03, 6.72e-03, 7.91e-03, 6.75e-03, 7.86e-03, 7.02e-03, 6.70e-03, 7.99e-03, 7.82e-03,
    6.63e-03, 7.19e-03, 7.05e-03, 6.91e-03, 6.88e-03, 6.80e-03, 6.35e-03, 6.67e-03, 8.07e-03, 6.70e-03,
    5.65e-03, 6.35e-03, 5.77e-03, 6.52e-03, 6.04e-03, 6.08e-03, 5.71e-03, 5.79e-03, 5.79e-03, 5.90e-03,
    5.19e-03, 5.55e-03, 5.65e-03, 5.81e-03, 6.16e-03, 6.03e-03, 5.97e-03, 5.99e-03, 6.23e-03, 6.19e-03,
    5.53e-03
]
test_losses = [
    3.77e-01, 2.73e-01, 1.83e-01, 1.74e-01, 1.53e-01, 1.08e-01, 1.03e-01, 9.86e-02, 8.37e-02, 7.62e-02,
    7.57e-02, 7.64e-02, 7.94e-02, 7.39e-02, 6.94e-02, 6.84e-02, 7.04e-02, 6.40e-02, 7.08e-02, 6.53e-02,
    6.36e-02, 5.78e-02, 6.67e-02, 6.01e-02, 7.06e-02, 6.14e-02, 5.35e-02, 5.85e-02, 6.23e-02, 5.38e-02,
    4.84e-02, 4.75e-02, 4.53e-02, 4.65e-02, 5.37e-02, 5.15e-02, 4.23e-02, 5.35e-02, 4.27e-02, 4.40e-02,
    5.67e-02, 4.43e-02, 3.92e-02, 4.00e-02, 3.92e-02, 3.95e-02, 4.16e-02, 4.31e-02, 3.72e-02, 3.84e-02,
    3.84e-02, 3.51e-02, 3.41e-02, 4.08e-02, 3.42e-02, 3.90e-02, 3.93e-02, 3.42e-02, 3.52e-02, 3.70e-02,
    3.32e-02, 3.67e-02, 3.23e-02, 3.26e-02, 3.26e-02, 3.46e-02, 3.28e-02, 3.11e-02, 3.31e-02, 3.20e-02,
    3.26e-02, 3.20e-02, 3.30e-02, 3.00e-02, 3.06e-02, 3.70e-02, 3.05e-02, 3.05e-02, 3.26e-02, 3.27e-02,
    2.97e-02, 3.03e-02, 2.97e-02, 3.13e-02, 3.06e-02, 3.29e-02, 3.16e-02, 3.10e-02, 2.87e-02, 2.97e-02,
    2.96e-02, 2.99e-02, 3.05e-02, 2.88e-02, 2.97e-02, 3.01e-02, 2.83e-02, 2.84e-02, 2.91e-02, 2.84e-02,
    3.23e-02, 2.84e-02, 2.87e-02, 3.09e-02, 2.77e-02, 2.90e-02, 2.81e-02, 2.81e-02, 2.90e-02, 2.96e-02,
    2.86e-02, 2.80e-02, 2.98e-02, 2.88e-02, 2.77e-02, 2.72e-02, 2.82e-02, 2.91e-02, 2.86e-02, 2.79e-02,
    2.82e-02, 2.71e-02, 2.82e-02, 2.71e-02, 2.74e-02, 2.71e-02, 2.87e-02, 2.78e-02, 2.75e-02, 2.72e-02,
    2.75e-02, 2.86e-02, 2.81e-02, 2.79e-02, 2.75e-02, 2.87e-02, 2.76e-02, 2.82e-02, 2.79e-02, 2.81e-02,
    2.78e-02
]
# Plotting the epoch-loss curve
plt.figure(figsize=(12, 6))

plt.plot(steps, train_losses, label='Train Loss', color='blue')
plt.plot(steps, test_losses, label='Test Loss', color='red')

plt.title('Epoch-Loss Curve(Steps 10000 to 150000)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Annotate the best model
best_model_step = 140000
best_train_loss = train_losses[-11]
best_test_loss = test_losses[-11]
plt.annotate(f'Best Model\nTrain Loss: {best_train_loss:.2e}\nTest Loss: {best_test_loss:.2e}',
             xy=(best_model_step, best_test_loss), xytext=(best_model_step, best_test_loss + 0.01),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.yscale('log')  # Use log scale for the y-axis to better visualize the losses
plt.savefig('newDarcy.png')
plt.show()