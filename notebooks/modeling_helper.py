from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def dt_plot_surface(X_train, X_test, y_train, y_test, max_depth=None, max_leaf_nodes=None, min_impurity_decrease=0):
    x_train_min, x_train_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
    y_train_min, y_train_max = X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1
    x_test_min, x_test_max = X_test[:, 0].min() - 0.1, X_test[:, 0].max() + 0.1
    y_test_min, y_test_max = X_test[:, 1].min() - 0.1, X_test[:, 1].max() + 0.1
    
    x_min = min(x_train_min, x_test_min)
    x_max = max(x_train_max, x_test_max)
    y_min = min(y_train_min, y_test_min)
    y_max = max(y_train_max, y_test_max)
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease
    )
    model.fit(X_train, y_train)
    z = model.predict(X_grid)
    z = z.reshape(xx.shape)

    # Plot
    plt.figure()
    plt.contourf(xx, yy, z, cmap=ListedColormap(["blue", "red", "green"]), alpha=.2)
    plt.scatter(X_train[:, 0], X_train[:, 1], cmap=ListedColormap(["blue", "red", "green"]), s=20, c=y_train)
    plt.scatter(X_test[:, 0], X_test[:, 1], cmap=ListedColormap(["blue", "red", "green"]), s=20, c=y_test, marker='x')
    
    plt.title(f"Max Depth: {max_depth}, Max Leaves: {max_leaf_nodes}, Min Impurity Decrease: {min_impurity_decrease}")
    plt.show()
    
    # Print stats
    print(f"^^^ Train accuracy: {round(model.score(X_train, y_train), 2)}, Test accuracy: {round(model.score(X_test, y_test), 2)} ^^^")


def et_plot_surface(X_train, X_test, y_train, y_test, max_depth=None, n_estimators=100):
    x_train_min, x_train_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
    y_train_min, y_train_max = X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1
    x_test_min, x_test_max = X_test[:, 0].min() - 0.1, X_test[:, 0].max() + 0.1
    y_test_min, y_test_max = X_test[:, 1].min() - 0.1, X_test[:, 1].max() + 0.1
    
    x_min = min(x_train_min, x_test_min)
    x_max = max(x_train_max, x_test_max)
    y_min = min(y_train_min, y_test_min)
    y_max = max(y_train_max, y_test_max)
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict
    model = ExtraTreesClassifier(
        max_depth=max_depth, 
        n_estimators=n_estimators,
        random_state=32,
        n_jobs=-1
    )
    model.fit(X_train, y_train.ravel())
    z = model.predict(X_grid)
    z = z.reshape(xx.shape)

    # Plot
    plt.figure()
    plt.contourf(xx, yy, z, cmap=ListedColormap(["blue", "red", "green"]), alpha=.2)
    plt.scatter(X_train[:, 0], X_train[:, 1], cmap=ListedColormap(["blue", "red", "green"]), s=20, c=y_train)
    plt.scatter(X_test[:, 0], X_test[:, 1], cmap=ListedColormap(["blue", "red", "green"]), s=20, c=y_test, marker='x')
    
    plt.title(f"Max Depth: {max_depth}, Number of Estimators (Trees): {n_estimators}")
    plt.show()
    
    # Print stats
    print(f"^^^ Train accuracy: {round(model.score(X_train, y_train), 2)}, Test accuracy: {round(model.score(X_test, y_test), 2)} ^^^")


def bag_plot_surface(X_train, X_test, y_train, y_test, max_samples=1.0, n_estimators=10):
    x_train_min, x_train_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
    y_train_min, y_train_max = X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1
    x_test_min, x_test_max = X_test[:, 0].min() - 0.1, X_test[:, 0].max() + 0.1
    y_test_min, y_test_max = X_test[:, 1].min() - 0.1, X_test[:, 1].max() + 0.1
    
    x_min = min(x_train_min, x_test_min)
    x_max = max(x_train_max, x_test_max)
    y_min = min(y_train_min, y_test_min)
    y_max = max(y_train_max, y_test_max)
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict
    model = BaggingClassifier(
        max_samples=max_samples,
        n_estimators=n_estimators,
        random_state=32,
        n_jobs=-1
    )
    model.fit(X_train, y_train.ravel())
    z = model.predict(X_grid)
    z = z.reshape(xx.shape)

    # Plot
    plt.figure()
    plt.contourf(xx, yy, z, cmap=ListedColormap(["blue", "red", "green"]), alpha=.2)
    plt.scatter(X_train[:, 0], X_train[:, 1], cmap=ListedColormap(["blue", "red", "green"]), s=20, c=y_train)
    plt.scatter(X_test[:, 0], X_test[:, 1], cmap=ListedColormap(["blue", "red", "green"]), s=20, c=y_test, marker='x')
    
    plt.title(f"Max Samples: {max_samples}, Number of Estimators (Trees): {n_estimators}")
    plt.show()
    
    # Print stats
    print(f"^^^ Train accuracy: {round(model.score(X_train, y_train), 2)}, Test accuracy: {round(model.score(X_test, y_test), 2)} ^^^")