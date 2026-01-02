import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.kernel = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.bias = np.zeros(out_channels)
        
        self.d_kernel = None
        self.d_bias = None
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        batch_size, h_in, w_in, c_in = x.shape
        
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (self.padding, self.padding), 
                                   (self.padding, self.padding), (0, 0)), mode='constant')
        else:
            x_padded = x
        
        h_out = (h_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        output = np.zeros((batch_size, h_out, w_out, self.out_channels))
        
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        
                        output[b, i, j, oc] = np.sum(
                            x_padded[b, h_start:h_end, w_start:w_end, :] * self.kernel[oc].transpose(1, 2, 0)
                        ) + self.bias[oc]
        
        return output

    def backward(self, dout):
        batch_size, h_out, w_out, c_out = dout.shape
        x_padded = self.last_input
        
        if self.padding > 0:
            x_padded = np.pad(self.last_input, ((0, 0), (self.padding, self.padding), 
                                                (self.padding, self.padding), (0, 0)), mode='constant')
        
        _, h_padded, w_padded, _ = x_padded.shape
        
        self.d_kernel = np.zeros_like(self.kernel)
        self.d_bias = np.zeros_like(self.bias)
        dx_padded = np.zeros_like(x_padded)
        
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        
                        self.d_kernel[oc] += x_padded[b, h_start:h_end, w_start:w_end, :].transpose(2, 0, 1) * dout[b, i, j, oc]
                        dx_padded[b, h_start:h_end, w_start:w_end, :] += self.kernel[oc].transpose(1, 2, 0) * dout[b, i, j, oc]
                
                self.d_bias[oc] += np.sum(dout[b, :, :, oc])
        
        if self.padding > 0:
            dx = dx_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dx = dx_padded
        
        return dx


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.last_input = None
        self.max_indices = None

    def forward(self, x):
        self.last_input = x
        batch_size, h_in, w_in, c_in = x.shape
        
        h_out = (h_in - self.pool_size) // self.stride + 1
        w_out = (w_in - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, h_out, w_out, c_in))
        self.max_indices = []
        
        for b in range(batch_size):
            for c in range(c_in):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        
                        pool_region = x[b, h_start:h_end, w_start:w_end, c]
                        max_val = np.max(pool_region)
                        output[b, i, j, c] = max_val
                        
                        max_pos = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                        self.max_indices.append((b, c, i, j, h_start + max_pos[0], w_start + max_pos[1]))
        
        return output

    def backward(self, dout):
        batch_size, h_out, w_out, c_out = dout.shape
        dx = np.zeros_like(self.last_input)
        
        for b, c, i, j, h_idx, w_idx in self.max_indices:
            if c >= dx.shape[3]:
                print(f"Error: c={c}, dx.shape[3]={dx.shape[3]}, max_indices entry: {(b, c, i, j, h_idx, w_idx)}")
                print(f"dout shape: {dout.shape}, dx shape: {dx.shape}")
                raise IndexError(f"Channel index {c} out of bounds for axis 3 with size {dx.shape[3]}")
            dx[b, h_idx, w_idx, c] += dout[b, i, j, c]
        
        return dx


class Dense:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        self.w = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)
        
        self.d_w = None
        self.d_b = None
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        return np.dot(x, self.w) + self.b

    def backward(self, dout):
        self.d_w = np.dot(self.last_input.T, dout)
        self.d_b = np.sum(dout, axis=0)
        dx = np.dot(dout, self.w.T)
        return dx


def relu(x):
    return np.maximum(0, x)


def relu_backward(dout, x):
    dx = dout * (x > 0)
    return dx


def softmax(x):
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class CrossEntropyLoss:
    def __init__(self):
        self.last_pred = None
        self.last_true = None
        self.loss = None

    def forward(self, pred, true):
        self.last_pred = pred
        self.last_true = true
        pred_softmax = softmax(pred)
        epsilon = 1e-15
        pred_softmax = np.clip(pred_softmax, epsilon, 1 - epsilon)
        self.loss = -np.sum(true * np.log(pred_softmax)) / pred.shape[0]
        return self.loss

    def backward(self):
        pred_softmax = softmax(self.last_pred)
        dout = (pred_softmax - self.last_true) / self.last_pred.shape[0]
        return dout


class CNN:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        
        self.conv1 = Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        
        self.conv2 = Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = MaxPool2D(pool_size=2, stride=2)
        
        self.fc1 = Dense(in_features=32 * 7 * 7, out_features=128)
        self.fc2 = Dense(in_features=128, out_features=10)
        
        self.relu1_out = None
        self.relu2_out = None
        self.relu_fc1_out = None

    def forward(self, x, training=True):
        x = self.conv1.forward(x)
        x = relu(x)
        self.relu1_out = x
        x = self.pool1.forward(x)
        
        x = self.conv2.forward(x)
        x = relu(x)
        self.relu2_out = x
        x = self.pool2.forward(x)
        
        x = x.reshape(x.shape[0], -1)
        
        x = self.fc1.forward(x)
        x = relu(x)
        self.relu_fc1_out = x
        
        x = self.fc2.forward(x)
        
        return x

    def backward(self, dout):
        dout = self.fc2.backward(dout)
        
        dout = relu_backward(dout, self.relu_fc1_out)
        dout = self.fc1.backward(dout)
        
        dout = dout.reshape(-1, 7, 7, 32)
        
        dout = self.pool2.backward(dout)
        
        dout = relu_backward(dout, self.relu2_out)
        dout = self.conv2.backward(dout)
        
        dout = self.pool1.backward(dout)
        
        dout = relu_backward(dout, self.relu1_out)
        dout = self.conv1.backward(dout)
        
        return dout

    def update_weights(self):
        self.conv1.kernel -= self.lr * self.conv1.d_kernel
        self.conv1.bias -= self.lr * self.conv1.d_bias
        
        self.conv2.kernel -= self.lr * self.conv2.d_kernel
        self.conv2.bias -= self.lr * self.conv2.d_bias
        
        self.fc1.w -= self.lr * self.fc1.d_w
        self.fc1.b -= self.lr * self.fc1.d_b
        
        self.fc2.w -= self.lr * self.fc2.d_w
        self.fc2.b -= self.lr * self.fc2.d_b

    def train(self, x_train, y_train, epochs, batch_size, x_test, y_test):
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        
        criterion = CrossEntropyLoss()
        num_samples = x_train.shape[0]
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            
            indices = np.random.permutation(num_samples)
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                pred = self.forward(x_batch, training=True)
                loss = criterion.forward(pred, y_batch)
                epoch_loss += loss * x_batch.shape[0]
                
                pred_labels = np.argmax(pred, axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                correct += np.sum(pred_labels == true_labels)
                
                dout = criterion.backward()
                self.backward(dout)
                self.update_weights()
            
            avg_train_loss = epoch_loss / num_samples
            train_acc = correct / num_samples
            
            test_loss, test_acc = self.evaluate(x_test, y_test, criterion)
            
            train_losses.append(avg_train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            
            print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        total_time = time.time() - start_time
        avg_time = total_time / epochs
        
        print(f"\n训练总时间: {total_time:.2f}秒")
        print(f"每轮平均训练时间: {avg_time:.2f}秒")
        
        return train_losses, train_accs, test_losses, test_accs

    def evaluate(self, x_test, y_test, criterion):
        pred = self.forward(x_test, training=False)
        loss = criterion.forward(pred, y_test)
        
        pred_labels = np.argmax(pred, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        acc = np.mean(pred_labels == true_labels)
        
        return loss, acc

    def predict(self, x):
        pred = self.forward(x, training=False)
        pred_labels = np.argmax(pred, axis=1)
        return pred_labels


def load_mnist_data(num_train=1000, num_test=100):
    print("Loading MNIST dataset...")
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    x_train_list = []
    y_train_list = []
    for img, label in mnist_train:
        x_train_list.append(img.numpy())
        y_train_list.append(label)
        if len(x_train_list) >= num_train:
            break
    
    x_test_list = []
    y_test_list = []
    for img, label in mnist_test:
        x_test_list.append(img.numpy())
        y_test_list.append(label)
        if len(x_test_list) >= num_test:
            break
    
    x_train = np.array(x_train_list)
    y_train = np.array(y_train_list)
    x_test = np.array(x_test_list)
    y_test = np.array(y_test_list)
    
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)
    
    y_train_onehot = np.zeros((y_train.shape[0], 10))
    y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1
    
    y_test_onehot = np.zeros((y_test.shape[0], 10))
    y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1
    
    print(f"Dataset loaded successfully!")
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train_onehot.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test_onehot.shape}")
    
    return x_train, y_train_onehot, x_test, y_test_onehot


def visualize_predictions(model, x_test, y_test, num_samples=10):
    pred_labels = model.predict(x_test)
    true_labels = np.argmax(y_test, axis=1)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(x_test[i, :, :, 0], cmap='gray')
        axes[i].axis('off')
        
        pred = pred_labels[i]
        true = true_labels[i]
        
        if pred == true:
            color = 'green'
            title = f'True: {true}, Pred: {pred}'
        else:
            color = 'red'
            title = f'True: {true}, Pred: {pred}'
        
        axes[i].set_title(title, color=color)
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    print("预测结果可视化已保存为 'predictions.png'")
    plt.close()


def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("训练曲线已保存为 'training_history.png'")
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)
    
    x_train, y_train, x_test, y_test = load_mnist_data(num_train=2000, num_test=200)
    
    model = CNN(learning_rate=0.01)
    
    epochs = 5
    batch_size = 128
    
    train_losses, train_accs, test_losses, test_accs = model.train(
        x_train, y_train, epochs, batch_size, x_test, y_test
    )
    
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    
    visualize_predictions(model, x_test, y_test, num_samples=10)
    
    print("\nExperiment completed!")
