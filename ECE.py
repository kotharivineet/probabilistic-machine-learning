import numpy as np

def ECE(pred, var, truth):
    '''
    Returns Expected calibration error, Accuracy, Confidence.
    '''

    sigma = np.sqrt(var)

    half_sigma_upper = pred + 0.5*sigma
    half_sigma_lower = pred - 0.5*sigma
    one_sigma_upper = pred + sigma
    one_sigma_lower = pred - sigma
    onehalf_sigma_upper = pred + 1.5*sigma
    onehalf_sigma_lower = pred - 1.5*sigma
    two_sigma_upper = pred + 2*sigma
    two_sigma_lower = pred - 2*sigma
    twohalf_sigma_upper = pred + 2.5*sigma
    twohalf_sigma_lower = pred - 2.5*sigma
    three_sigma_upper = pred + 3*sigma
    three_sigma_lower = pred - 3*sigma

    half_sigma = 0
    one_sigma = 0
    onehalf_sigma = 0
    two_sigma = 0
    twohalf_sigma = 0
    three_sigma = 0
    n_pred = len(pred)

    for i in range(n_pred):

        if truth[i] >= half_sigma_lower[i] and truth[i] <= half_sigma_upper[i]:
            half_sigma += 1

        if truth[i] >= one_sigma_lower[i] and truth[i] <= one_sigma_upper[i]:
            one_sigma += 1

        if truth[i] >= onehalf_sigma_lower[i] and truth[i] <= onehalf_sigma_upper[i]:
            onehalf_sigma += 1

        if truth[i] >= two_sigma_lower[i] and truth[i] <= two_sigma_upper[i]:
            two_sigma += 1

        if truth[i] >= twohalf_sigma_lower[i] and truth[i] <= twohalf_sigma_upper[i]:
            twohalf_sigma += 1

        if truth[i] >= three_sigma_lower[i] and truth[i] <= three_sigma_upper[i]:
            three_sigma += 1

    Acc = np.asarray([0,
        half_sigma/n_pred,
        one_sigma/n_pred,
        onehalf_sigma/n_pred,
        two_sigma/n_pred,
        twohalf_sigma/n_pred,
        three_sigma/n_pred])

    actual_half_sigma = 0.3829
    actual_one_sigma = 0.6826
    actual_onehalf_sigma = 0.8663
    actual_two_sigma = 0.9544
    actual_twohalf_sigma = 0.9875
    actual_three_sigma = 0.9973
    
    Conf = np.asarray([0, actual_half_sigma, actual_one_sigma, actual_onehalf_sigma, actual_two_sigma, actual_twohalf_sigma, actual_three_sigma])

    ece = np.sum(np.abs(Acc[1:7]-Conf[1:7]))/6

    return ece, Acc, Conf



# import numpy as np
# import matplotlib.pyplot as plt

# N = 1000
# mu = np.sin(np.linspace(0,100,N))
# # mu = np.random.normal(0,1,N)
# truth = np.ones(N)*mu
# pred = truth + np.random.normal(0,1,N)
# var = 1*np.ones(N)

# ece, Acc, Conf = ECE(pred, var, truth)
# print(ece)
# plt.plot(Conf, Acc)
# plt.plot([0,1],[0,1],linestyle='--')
# plt.xlabel('Conf')
# plt.ylabel('Acc')
# plt.title('Accurate Model')

# plt.figure()

# plt.plot(np.linspace(0,100,N), pred)
# plt.plot(np.linspace(0,100,N), truth)
# plt.show()
