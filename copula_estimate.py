import numpy as np

from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.autograd import Variable
import math


class GaussianCopula(torch.nn.Module):
    r"""
    The Gaussian (Normal) copula. It is elliptical and symmetric which gives it nice analytical properties. The
    Gaussian copula is determined entirely by its correlation matrix.


    A Gaussian copula is fined as

    .. math::

        C_\Sigma (u_1, \dots, u_d) = \Phi_\Sigma (N^{-1} (u_1), \dots, N^{-1} (u_d))

    where :math:`\Sigma` is the covariance matrix which is the parameter of the Gaussian copula and
    :math:`N^{-1}` is the quantile (inverse cdf) function
    """

    def __init__(self, dim=2):
        """
        Creates a Gaussian copula instance

        Parameters
        ----------
        dim: int, optional
            Dimension of the copula
        """
        super(GaussianCopula, self).__init__()
        n = sum(range(dim))
        self.dim = int(dim)
        self._rhos = np.zeros(n)
        self._bounds = np.repeat(-1., n), np.repeat(1., n)
        self.sig = nn.Sigmoid()
        self.off_diagonal_val = nn.Parameter(torch.rand(int(self.dim*(self.dim-1)/2)).cuda())
        self.diagonal_val = nn.Parameter(torch.ones(self.dim).cuda() + torch.ones(self.dim).cuda()*torch.rand(self.dim).cuda())
        

    def forward(self, data, margins='Normal', hyperparam=None):
        """
        Fit the copula with specified data

        Parameters
        ----------
        data: ndarray
            Array of data used to fit copula. Usually, data should be the pseudo observations

        marginals : numpy array
		The marginals distributions. Use scipy.stats distributions or equivalent that requires pdf and cdf functions according to rv_continuous class from scipy.stat.

	    hyper_param : numpy array
		The hyper-parameters for each marginal distribution. Use None when the hyper-parameter is unknow and must be estimated.
		"""

        data = self.pobs(data)
        if data.ndim != 2:
            raise ValueError('Data must be a matrix of dimension (n x d)')
        elif self.dim != data.shape[1]:
            raise ValueError('Dimension of data does not match copula')
        data = torch.from_numpy(data).float()
        # print('transformed data:',data)
        return self.mle(data, margins, hyperparam)


    def pobs(self, data, ties='average'):
        """
        Compute the pseudo-observations for the given data matrix

        Parameters
        ----------
        data: {array_like, DataFrame}
            Random variates to be converted to pseudo-observations

        ties: { 'average', 'min', 'max', 'dense', 'ordinal' }, optional
            String specifying how ranks should be computed if there are ties in any of the coordinate samples

        Returns
        -------
        ndarray
            matrix or vector of the same dimension as `data` containing the pseudo observations
        """
        return self.pseudo_obs(data, ties)

    def pseudo_obs(self, data, ties='average'):
        """
        Compute the pseudo-observations for the given data matrix

        Parameters
        ----------
        data: (N, D) ndarray
            Random variates to be converted to pseudo-observations

        ties: str, optional
            The method used to assign ranks to tied elements. The options are 'average', 'min', 'max', 'dense'
            and 'ordinal'.
            'average': The average of the ranks that would have been assigned to all the tied values is assigned to each
                value.
            'min': The minimum of the ranks that would have been assigned to all the tied values is assigned to each
                value. (This is also referred to as "competition" ranking.)
            'max': The maximum of the ranks that would have been assigned to all the tied values is assigned to each value.
            'dense': Like 'min', but the rank of the next highest element is assigned the rank immediately after those
                assigned to the tied elements. 'ordinal': All values are given a distinct rank, corresponding to
                the order that the values occur in `a`.

        Returns
        -------
        ndarray
            matrix or vector of the same dimension as `data` containing the pseudo observations
        """
        return self.rank_data(data, 1, ties) / (len(data) + 1)

    def rank_data(self, obs, axis=0, ties='average'):
        """
        Assign ranks to data, dealing with ties appropriately. This function works on core as well as vectors

        Parameters
        ----------
        obs: ndarray
            Data to be ranked. Can only be 1 or 2 dimensional.
        axis: {0, 1}, optional
            The axis to perform the ranking. 0 means row, 1 means column.
        ties: str, default 'average'
            The method used to assign ranks to tied elements. The options are 'average', 'min', 'max', 'dense'
            and 'ordinal'.
            'average': The average of the ranks that would have been assigned to all the tied values is assigned to each
                value.
            'min': The minimum of the ranks that would have been assigned to all the tied values is assigned to each
                value. (This is also referred to as "competition" ranking.)
            'max': The maximum of the ranks that would have been assigned to all the tied values is assigned to each value.
            'dense': Like 'min', but the rank of the next highest element is assigned the rank immediately after those
                assigned to the tied elements. 'ordinal': All values are given a distinct rank, corresponding to
                the order that the values occur in `a`.

        Returns
        -------
        ndarray
            matrix or vector of the same dimension as X containing the pseudo observations
        """
        obs = np.asarray(obs)
        ties = ties.lower()
        assert obs.ndim in (1, 2), "Data can only be 1 or 2 dimensional"

        if obs.ndim == 1:
            return stats.rankdata(obs, ties)
        elif obs.ndim == 2:
            if axis == 0:
                return np.array([stats.rankdata(obs[i, :], ties) for i in range(obs.shape[0])])
            elif axis == 1:
                return np.array([stats.rankdata(obs[:, i], ties) for i in range(obs.shape[1])]).T
            else:
                raise ValueError("No axis named 3 for object type {0}".format(type(obs)))

    def get_R(self):
        '''

        :return:
        '''
        idx = 0
        L = torch.zeros(self.dim, self.dim).cuda()
        off_diag = torch.zeros(self.dim, self.dim).cuda()

        for j in range(self.dim):
            for i in range(j):
                off_diag[j, i] = torch.tanh(self.off_diagonal_val[idx])
                idx += 1

        for i in range(self.dim):
            for j in range(i+1):
                if i == j:
                    # print('sig diagoal:', self.sig(self.diagonal_val[i]))
                    L[i, j] = self.sig(self.diagonal_val[i]) + torch.tensor([1.0]).cuda()
                else:
                    L[i, j] = off_diag[i,j]

        L = F.normalize(L, p=2, dim=0)
        # print('L:', L)
        cov = torch.mm(L, L.t())

        return cov

    def pdf_param(self, x):
        # self._check_dimension(x)
        '''

        :param x: data numpy  n*d
        :param R: flattened correlation value, not include the redundancy and diagonal value shape: (d*(d-1))/2
        :return:
        '''

        # print('R:',R)
        # print('x:',x)
        norm = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        u = norm.icdf(x)


        # print('shape u:', u.shape)

        cov = self.get_R().cuda()

        if self.dim == 2:
            RDet = cov[0,0] * cov[1,1] - cov[0,1] ** 2
            RInv = 1. / RDet * torch.from_numpy(np.asarray([[cov[1,1], -cov[0,1]], [-cov[0,1], cov[0,0]]]))
        else:
            RDet = torch.det(cov)
            RInv = torch.inverse(cov)
        u = u.unsqueeze(0).cuda()


        # print('u shape', u.shape)  #d
        I = torch.eye(self.dim).cuda()
        # print('u cuda:', u.is_cuda)
        # print('cov cuda:', cov.is_cuda)
        # print('I cuda:', I.is_cuda)
        res = RDet ** (-0.5) * torch.exp(-0.5 * torch.mm(torch.mm(u,(RInv - I)),u.permute(1,0))).cuda()
        # print('res:', res)
        if res.data == 0.0:
            print('RDet:', RDet)
            print('RInv shape', RInv.shape)
        if math.isnan(res.data):
            print('self.diagonal:', self.diagonal_val)
            print('self.non_diagonal:', self.off_diagonal_val)
            print('RDet:', RDet)

            print('RInv:', RInv)
            print('cov:', cov)
            print('u:', u)
            return

        return res


    def mle(self, X, marginals, hyper_param):
        """
        Computes the MLE on specified data.

        Parameters
        ----------
        copula : Copula
            The copula.
        X : numpy array (of size n * copula dimension)
            The data to fit.
        marginals : numpy array
            The marginals distributions. Use scipy.stats distributions or equivalent that requires pdf and cdf functions according to rv_continuous class from scipy.stat.
        hyper_param : numpy array
            The hyper-parameters for each marginal distribution. Use None when the hyper-parameter is unknow and must be estimated.

        Returns
        -------
        loss : copula loss
        """
        hyperOptimizeParams = hyper_param

        n, d = X.shape

        # The global log-likelihood to maximize
        def log_likelihood():

            lh = 0

            marginCDF = torch.zeros(n,d)
            if marginals == 'Normal':
                for j in range(d):
                    norm = Normal(hyperOptimizeParams[j]['loc'], hyperOptimizeParams[j]['scale'])
                    marginCDF[:,j] = norm.cdf(X[:, j]).cuda()
                    # print('marginCDF[:,j] shape:', marginCDF[:,j].shape)
                    # print('marginCDF[:,j]:', marginCDF[:,j])
                    idx = np.argwhere(marginCDF[:,j] == 1.0)
                    if idx.nelement() != 0:
                        for i in range(idx.shape[1]):
                            marginCDF[idx[0,i].item(),j] -= 1e-2




            # The first member : the copula's density
            for i in range(n):
                pdf_val = self.pdf_param(marginCDF[i, :])
                lh += torch.log(pdf_val if pdf_val.data !=0.0 else torch.tensor([[1e-5]]).cuda())

            # The second member : sum of PDF
            # print("OK")
            # print('first lh:', lh)
            for j in range(d):
                norm = Normal(hyperOptimizeParams[j]['loc'], hyperOptimizeParams[j]['scale'])
                lh += (norm.log_prob(X[:, j])).sum().cuda()

            # print('lh:', lh)
            return lh


        loss = -log_likelihood()
        return loss



# if __name__ == '__main__':
#     data = np.random.random((4,3))
#     print('data:', data)
#     data[:,-1] = [1,1,1,1]
#     print('data:',data)
#     gc = GaussianCopula(dim=3)
#     res, param = gc.fit(data, margins=[stats.norm, stats.norm,stats.norm], hyperparam=[{'loc':0, 'scale':1.2},{'loc':0.2, 'scale':1.2},{'loc':0.2, 'scale':1.0}])
#
#     print('optimizeResult:', res)
#     print('estimatedParams:', param)