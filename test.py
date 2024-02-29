from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    x = x - np.mean(x, axis=0)

    return x

def get_covariance(dataset):
    return np.dot(np.transpose(dataset), dataset) / (dataset.shape[0] - 1)

def get_eig(S, m):
    eigenValues, eigenVectors = eigh(S, subset_by_index=[S.shape[0] - m, S.shape[0] - 1])
    return np.diag(eigenValues[::-1]), eigenVectors[:, ::-1]

def get_eig_prop(S, prop):
    totalVariance = S.trace()
    requiredVariance = totalVariance * prop
    eigenValues, eigenVectors = eigh(S, subset_by_value=[requiredVariance, np.inf])
    return np.diag(eigenValues[::-1]), eigenVectors[:, ::-1]

def project_image(image, U):
    return (np.dot(U, np.dot(np.transpose(U), image)))

def display_image(orig, proj):
    original = orig.reshape(32, 32)
    projected = proj.reshape(32, 32)
    original = np.rot90(original, 3)
    projected = np.rot90(projected, 3)
    # Please use the format below to ensure grading consistency
    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    ax1.set_title('Original')
    ax2.set_title('Projection')
    colorbar1 = ax1.imshow(original, aspect='equal')
    colorbar2 = ax2.imshow(projected, aspect='equal')
    fig.colorbar(colorbar1, ax=ax1)
    fig.colorbar(colorbar2, ax=ax2)
    return fig, ax1, ax2

def main():
    x = load_and_center_dataset('YaleB_32x32.npy')
    S = get_covariance(x)
    Lambda, U = get_eig_prop(S, 0.07)
    for i in range(1500,1510):
        projected = project_image(x[i], U)
        display_image(x[i], projected)
        plt.show()


if __name__ == "__main__":
    main()
