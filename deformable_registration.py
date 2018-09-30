import numpy as np


class deformable_registration(object):
    def __init__(self, X, Y, _lambda=None, alpha=None, beta=None, sigma2=None, maxIterations=100, tolerance=0.001, w=0):
        if X.shape[1] != Y.shape[1]:
            raise 'Both point clouds must have the same number of dimensions!'

        self.X = X
        self.Y = Y
        self.TY = Y
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape

        self._lambda = 2 if _lambda is None else _lambda
        self.beta = 2 if beta is None else beta
        self.alpha = 0.95**2 if alpha is None else alpha


        self.W = np.zeros((self.M, self.D))
        self.G = np.zeros((self.M, self.M))
        self.sigma2 = sigma2
        self.iteration = 0
        self.maxIterations = maxIterations
        self.tolerance = tolerance
        self.w = w
        self.q = 0
        self.err = 0

    def register(self, callback):
        self.initialize()

        while self.iteration < self.maxIterations and self.err > self.tolerance:
            self.iterate()
            if callback:
                callback(iteration=self.iteration, error=self.err, X=self.X, Y=self.TY)

        return self.TY, np.dot(self.G, self.W)

    def iterate(self):
        self.EStep()
        self.MStep()
        self.iteration += 1

    def MStep(self):
        self.updateTransform()
        self.transformPointCloud()
        self.updateVariance()

    def updateTransform(self):
        # print(self.P1)
        # print(self.G)
        A = np.dot(np.diag(self.P1), self.G) + self._lambda * self.sigma2 * np.eye(self.M)
        B = np.dot(self.P, self.X) - np.dot(np.diag(self.P1), self.Y)
        self.W = np.linalg.solve(A, B)
        # annealing
        # self.sigma2 = self.sigma2 * self.alpha
        # print(self.sigma2)

    def transformPointCloud(self, Y=None):
        if Y is None:
            self.TY = self.Y + np.dot(self.G, self.W)
            return
        else:
            return Y + np.dot(self.G, self.W)

    def updateVariance(self):
        qprev = self.sigma2

        xPx = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.X, self.X), axis=1))
        # print(self.X.shape)
        # yPy = np.dot(np.transpose(self.P1), np.sum(np.multiply(self.Y, self.Y), axis=1))
        # trPXY = np.sum(np.multiply(self.Y, np.dot(self.P, self.X)))
        yPy = np.dot(np.transpose(self.P1), np.sum(np.multiply(self.TY, self.TY), axis=1))
        trPXY = np.sum(np.multiply(self.TY, np.dot(self.P, self.X)))


        self.sigma2 = (xPx - 2*trPXY + yPy) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        self.err = np.abs(self.sigma2 - qprev)


    def initialize(self):
        if not self.sigma2:
            # print(self.X.shape)
            XX = np.reshape(self.X, (1, self.N, self.D))
            # print(XX.shape)
            YY = np.reshape(self.Y, (self.M, 1, self.D))
            # print(YY)
            XX = np.tile(XX, (self.M, 1, 1))
            # print(XX)
            YY = np.tile(YY, (1, self.N, 1))
            # print(YY)
            diff = XX - YY
            err = np.multiply(diff, diff) # with dim: M X N X D
            # calculate euclidean distance between X and Y into err matrix

            self.sigma2 = np.sum(err) / (self.D * self.M * self.N)

        self.err = self.tolerance + 1
        self.q = -self.err - self.N * self.D / 2 * np.log(self.sigma2)
        self._makeKernel()

    def EStep(self):
        P = np.zeros((self.M, self.N))

        for i in range(0, self.M):
            diff = self.X - np.tile(self.TY[i, :], (self.N, 1))

            diff = np.multiply(diff, diff) # inplace square elements
            P[i, :] = P[i, :] + np.sum(diff, axis=1) # P here equals to distance^2
            # print(diff)
            # break


        c = (2 * np.pi * self.sigma2) ** (self.D / 2)
        c = c * self.w / (1 - self.w)
        c = c * self.M / self.N

        P = np.exp(-P / (2 * self.sigma2)) # turn into probability

        den = np.sum(P, axis=0)
        den = np.tile(den, (self.M, 1))
        den[den == 0] = np.finfo(float).eps # make it non-zero
        den += c

        self.P = np.divide(P, den) # normalize P
        # print(np.max(self.P, axis=1).argsort()[:10])
        # pp = np.array(P[128, :]).argsort()[-5:]
        # print(np.array(P[128, :])[pp])
        self.Pt1 = np.sum(self.P, axis=0) # sum of every point in X
        self.P1 = np.sum(self.P, axis=1)    # sum of every point in Y
        self.Np = np.sum(self.P1) # sum of all probs

    def _makeKernel(self):
        XX = np.reshape(self.Y, (1, self.M, self.D))
        YY = np.reshape(self.Y, (self.M, 1, self.D))
        XX = np.tile(XX, (self.M, 1, 1))
        YY = np.tile(YY, (1, self.M, 1))
        diff = XX - YY
        diff = np.multiply(diff, diff)
        # print(diff.shape)
        # calculate distance of every pair of points in Y
        diff = np.sum(diff, axis=2)
        # print(diff.shape)
        self.G = np.exp(-diff / (2 * self.beta)) # Just probability value (single, not continuous func)
