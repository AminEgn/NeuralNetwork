# standard
from time import sleep
import random
import os


###############################
# No validation on this class #
###############################

class Perceptron(object):
    """Perceptron"""

    def __init__(
            self,
            samples: list[list[float]],
            targets: list[int],
            learning_rate: float = 0.01,
            bias: float = 1.0,
            epoch_number: int = 1000,
    ):
        self.samples = samples
        self.targets = targets
        self.learning_rate = learning_rate
        self.bias = bias
        self.epoch_number = epoch_number
        self._initialize()

    def _initialize(self):
        """initialize essential data for training"""
        self.weights = [self.bias]
        # generate random weights
        for _ in range(len(self.samples[0])):
            self.weights.append(random.random())
        # add bias to every sample's list
        for sample in self.samples:
            sample.append(self.bias)

    def _validate_arguments(self):
        """validate init arguments"""

    @staticmethod
    def _activation(value: float):
        """activation function"""
        return 1 if value >= 0 else -1

    def _update_weights(self, sample, err):
        """update weight"""
        for pos, value in enumerate(sample):
            weight = self.weights[pos]
            new_weight = weight + self.learning_rate * err * value
            self.weights[pos] = new_weight

    @staticmethod
    def _cmd_output(epoch, weights):
        """output in command line"""
        # clean command line
        os.system("clear")
        print(
            f"Number of epoch: {epoch}",
            "\n",
            f"weights: {weights}"
        )

    def train(self):
        """training function"""
        epoch_try = 0
        while self.epoch_number >= 0:
            suitable_weights = True
            for index, sample in enumerate(self.samples):
                computed = 0
                target = self.targets[index]
                for pos, value in enumerate(sample):
                    computed += self.weights[pos] * value
                # check for activation of desire
                desire = self._activation(computed)
                if target != desire:
                    # update weights
                    err = target - desire
                    self._update_weights(sample, err)
                    suitable_weights = False

            # decrease number epoch
            self.epoch_number -= 1
            # increase number of try just for output
            epoch_try += 1
            self._cmd_output(epoch_try, self.weights)
            # set time sleep for better ux
            sleep(0.07)
            # suitable weight found !
            if suitable_weights:
                break


test_samples = [
    [-0.6508, 0.1097, 4.0009],
    [-1.4492, 0.8896, 4.4005],
    [2.0850, 0.6876, 12.0710],
    [0.2626, 1.1476, 7.7985],
    [0.6418, 1.0234, 7.0427],
    [0.2569, 0.6730, 8.3265],
    [1.1155, 0.6043, 7.4446],
    [0.0914, 0.3399, 7.0677],
    [0.0121, 0.5256, 4.6316],
    [-0.0429, 0.4660, 5.4323],
    [0.4340, 0.6870, 8.2287],
    [0.2735, 1.0287, 7.1934],
    [0.4839, 0.4851, 7.4850],
    [0.4089, -0.1267, 5.5019],
    [1.4391, 0.1614, 8.5843],
    [-0.9115, -0.1973, 2.1962],
    [0.3654, 1.0475, 7.4858],
    [0.2144, 0.7515, 7.1699],
    [0.2013, 1.0014, 6.5489],
    [0.6483, 0.2183, 5.8991],
    [-0.1147, 0.2242, 7.2435],
    [-0.7970, 0.8795, 3.8762],
    [-1.0625, 0.6366, 2.4707],
    [0.5307, 0.1285, 5.6883],
    [-1.2200, 0.7777, 1.7252],
    [0.3957, 0.1076, 5.6623],
    [-0.1013, 0.5989, 7.1812],
    [2.4482, 0.9455, 11.2095],
    [2.0149, 0.6192, 10.9263],
    [0.2012, 0.2611, 5.4631],
]

test_targets = [
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    1,
    1,
    -1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
]


if __name__ == "__main__":
    perc = Perceptron(test_samples, test_targets)
    perc.train()
