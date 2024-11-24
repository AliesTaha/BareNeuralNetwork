import numpy as np

class Loss:
    def calculate(self, output, y):
        self.sample_losses = self.forward(output, y)
        data_loss = np.mean(self.sample_losses)
        return data_loss

class Loss_Categorical_Cross_Entropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            confidences = y_pred_clipped[range(len(y_pred)), y_true]
        elif len(y_true.shape) == 2:
            y_hat = y_pred_clipped * y_true
            confidences = np.sum(y_hat, axis=1)
        print("confidences")
        print(confidences)
        loss = -np.log(confidences)
        return loss