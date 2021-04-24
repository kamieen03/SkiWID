#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def vectors_uniform(k):
    """Uniformly generates k vectors."""
    vectors = []
    for a in np.linspace(0, 2 * np.pi, k, endpoint=False):
        vectors.append(2 * np.array([np.sin(a), np.cos(a)]))
    return vectors


def visualize_transformation(A, vectors):
    """Plots original and transformed vectors for a given 2x2 transformation matrix A and a list of 2D vectors."""
    for i, v in enumerate(vectors):
        # Plot original vector.
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.008, color="blue", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0]/2 + 0.25, v[1]/2, "v{0}".format(i), color="blue")

        # Plot transformed vector.
        tv = A.dot(v)
        plt.quiver(0.0, 0.0, tv[0], tv[1], width=0.005, color="magenta", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(tv[0] / 2 + 0.25, tv[1] / 2, "v{0}'".format(i), color="magenta")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.margins(0.05)
    # Plot eigenvectors
    plot_eigenvectors(A)
    plt.show()


def visualize_vectors(vectors, color="green"):
    """Plots all vectors in the list."""
    for i, v in enumerate(vectors):
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.006, color=color, scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0] / 2 + 0.25, v[1] / 2, "eigv{0}".format(i), color=color)


def plot_eigenvectors(A):
    """Plots all eigenvectors of the given 2x2 matrix A."""
    # TODO: Zad. 4.1. Oblicz wektory własne A. Możesz wykorzystać funkcję np.linalg.eig
    _, eigvecs = np.linalg.eig(A)
    # TODO: Zad. 4.1. Upewnij się poprzez analizę wykresów, że rysowane są poprawne wektory własne (łatwo tu o pomyłkę).
    visualize_vectors(eigvecs.T)


def EVD_decomposition(A):
    # TODO: Zad. 4.2. Uzupełnij funkcję tak by obliczała rozkład EVD zgodnie z zadaniem.
    print('A:', '\n', A)
    vals, K = np.linalg.eig(A)
    if np.linalg.norm(K[:,0] - K[:,1]) < 1e-12:
        print('Eigenvectors are the same - no eigendecomposition')
        print()
        return
    Kinv = np.linalg.inv(K)
    L = np.diag(vals)
    print('K:', '\n', K)
    print('L:', '\n', L)
    print('K^-1:', '\n', Kinv)
    print()
    assert np.linalg.norm(K @ L @ Kinv - A) < 1e-12

def repeat(A, v):
    for n in range(30):     # '30' dobrane arbiralnie
        v = A@v
        v /= np.linalg.norm(v)
    return v

def check_if_all_eigen(eigvals, eigvecs):
    return abs(eigvals[0] - eigvals[1]) < 1e-12 and \
            np.linalg.norm(eigvecs[0] - eigvecs[1]) > 1e-12

def plot_attractors(A, vectors):
    # TODO: Zad. 4.3. Uzupełnij funkcję tak by generowała wykres z atraktorami.
    eigvals, eigvecs = np.linalg.eig(A)
    eigvecs = eigvecs.T
    if np.linalg.norm(eigvecs[0] - eigvecs[1]) < 1e-12:  # wektory własne są takie same
        attractors = [eigvecs[0], -eigvecs[0]]
        vals = [eigvals[0]] * 2
        colors = ['green', 'red']
    else:
        attractors = [*eigvecs, *-eigvecs]
        vals = [*eigvals, *eigvals]
        colors = ['green', 'red', 'orange', 'blue']

    for i, (e,a,c) in enumerate(zip(vals, attractors, colors)):  # plot attractors
        plt.quiver(0.0, 0.0, 3*a[0], 3*a[1], width=0.002, color=c, scale_units='xy', angles='xy', scale=1,
                zorder=5, headwidth=16)
        if i < len(attractors)//2:
            plt.text(1.1*3*a[0], 1.1*3*a[1], f'{e}', color="black")

    all_eigen = check_if_all_eigen(eigvals, eigvecs)
    for v in vectors:
        attracted = repeat(A,v)
        min_dist = np.inf
        color = 'black'
        if not all_eigen:
            for a,c in zip(attractors, colors):         # find the closest attractor
                dist = np.linalg.norm(a - attracted)
                if dist < min_dist:
                    min_dist = dist
                    color = c
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.002, color=color, scale_units='xy', angles='xy', scale=1,
                zorder=4, headwidth=8)

    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.margins(0.05)
    plt.show()


def show_eigen_info(A, vectors):
    EVD_decomposition(A)
    visualize_transformation(A, vectors)
    plot_attractors(A, vectors)


if __name__ == "__main__":
    vectors = vectors_uniform(k=16)

    A = np.array([[2, 0],
                  [0, 2]])
    show_eigen_info(A, vectors)


    A = np.array([[-1, 2],
                  [2, 1]])
    show_eigen_info(A, vectors)


    A = np.array([[3, 1],
                  [0, 2]])
    show_eigen_info(A, vectors)


    A = np.array([[2, -1],
                  [1, 4]])
    show_eigen_info(A, vectors)
