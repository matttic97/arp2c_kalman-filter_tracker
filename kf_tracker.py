import random
from kalman_filters import *
import utils
from ex2_utils import *
from ex4_utils import *


class KalmanFilterTracker(Tracker):

    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (int(region[2])+abs(int(region[2])%2-1), int(region[3])+abs(int(region[3])%2-1))
        self.nbins = 8
        template, _ = get_patch(image, self.position, self.size)
        self.kernel_g = create_epanechnik_kernel(template.shape[1], template.shape[0], 10)
        self.q = extract_histogram(template, self.nbins, self.kernel_g)

        self.motion_model = get_NCV_model(self.parameters.kf)
        x = np.array([random.uniform(self.position[0]-5, self.position[0]+5) for _ in range(self.parameters.N)])
        y = np.array([random.uniform(self.position[1]-5, self.position[1]+5) for _ in range(self.parameters.N)])
        self.particles = np.array([[x[i], y[i], 0, 0] for i in range(self.parameters.N)], dtype="float64") \
                         + sample_gauss([0, 0, 0, 0], self.motion_model[1], self.parameters.N)
        self.weights = np.ones(self.parameters.N)

    def track(self, image):
        # re s ample p a r t i c l e s
        weights_norm = self.weights / np.sum(self.weights)  # n o rm ali z e w ei g h t s
        weights_cumsumed = np.cumsum(weights_norm)  # cumul a ti ve d i s t r i b u t i o n
        rand_samples = np.random.rand(len(self.particles), 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)  # randomly s e l e c t N i n d i c e s
        particles = self.particles[sampled_idxs.flatten(), :]
        self.particles = np.matmul(particles, self.motion_model[0])

        for i in range(len(self.particles)):
            patch, mask = get_patch(image, self.particles[i][:2], self.size)
            p = extract_histogram(patch, self.nbins, self.kernel_g)
            d2 = utils.hellinger_distance(self.q, p)**2
            self.weights[i] = np.exp(-0.5 * (d2/self.parameters.sigma2))

        loc = np.sum(self.particles.T*self.weights, axis=1) / np.sum(self.weights)
        patch, mask = get_patch(image, loc[:2], self.size)
        h = extract_histogram(patch, self.nbins, self.kernel_g)
        self.q = (1-self.parameters.alpha)*self.q + self.parameters.alpha*h
        self.position = loc[:2]

        return [self.position[0] - self.size[0]/2, self.position[1]-self.size[1]/2, self.size[0], self.size[1]]


class KFParams():
    def __init__(self):
        self.N = 100
        self.alpha = 0.1
        self.sigma2 = 10
        self.kf = KalmanFilterParams(1, 0.001, 100)
