import numpy
import matplotlib.pyplot as plt

# Training: a9a/a9a / Test: a9a/a9a.t
datasetPath = []
datasetPath.append('a9a/a9a')
datasetPath.append('a9a/a9a.t')
maxIt = 5
accuracyStop = 0.01
convStop = 1.5


# Math function
def sigm(a):
    return 1. / (1. + numpy.exp(-a))


# Parser
def parser(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    x = numpy.zeros((len(lines), 124))
    y = numpy.ones(len(lines))
    i = 0
    for line in lines:
        data = line.strip().split()
        y[i] = (1. + int(data[0])) / 2.
        data = data[1:]
        for dataFeat in data:
            feat = dataFeat.split(':')
            x[i, int(feat[0])] = 1.0
        i += 1
    return x, y


# Accuracy calc
def accuracyCalc(ui, y):
    u = 0
    goodRes = 0
    while u < len(y):
        uiR = round(ui[u])
        if uiR == y[u]:
            goodRes += 1
        u += 1
    return (goodRes * 100) / u


# Print
def printRes(ui, y, i, logLikelihoodTab, lamb, normTab, accuracyTab):
    print("------------")
    print("i", i)
    print("lamb", lamb)
    print("logLikelihoodTab", logLikelihoodTab)
    print("accuracyTab", accuracyTab)
    print("normTab", normTab)
    u = 0
    goodRes = 0
    while u < len(y):
        uiR = round(ui[u])
        # print(uiR, "-", y[u])                      #Uncomment to print all results
        if uiR == y[u]:
            goodRes += 1
        u += 1
    print("Good result", (goodRes * 100) / u, "%")
    print("------------")


# Plot data
itFig = 0


def plotDataTrain(i, logLikelihoodTab, normTab, accuracyTab):
    plt.figure(figsize=(15, 5))
    plt.subplot(3, 1, 1)
    plt.plot(range(1, i + 1), logLikelihoodTab)
    plt.ylabel('Log-Likelihood')
    plt.xlabel('Iterations IRLS')
    plt.subplot(2, 1, 2)
    plt.plot(range(1, i + 1), normTab)
    plt.ylabel('Convergence, Norm')
    plt.xlabel('Iterations IRLS')
    plt.subplot(3, 1, 3)
    plt.plot(range(1, i + 1), accuracyTab)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Iterations IRLS')
    global itFig
    plt.savefig("figures/fig_train" + str(itFig))
    itFig += 1

def plotDataTest(lambTab, accuracyTab, iTab):
    plt.figure(figsize=(10, 5))
    plt.tight_layout()
    plt.subplot(1, 2, 1)
    plt.bar(lambTab, iTab)
    plt.ylabel('Iterations IRLS')
    plt.xlabel('Lambda')
    plt.subplot(1, 2, 2)
    plt.bar(lambTab, accuracyTab)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Lambda')
    global itFig
    plt.savefig("figures/fig_test" + str(itFig))
    itFig += 1


# LogLikehood
def logLikehood(ni, y):
    su = y * ni - numpy.log(1 + numpy.exp(ni))
    return su.sum()


# IRLS algo
def irls(x, y, lam):
    w = numpy.zeros(x.shape[1])
    w0 = numpy.log(numpy.average(y) / (1 - numpy.average(y)))
    logLikelihoodTab = []
    normTab = []
    accuracyTab = []
    wLast = 0
    for i in range(0, maxIt):
        ni = w0 + x.dot(w)
        logLikelihoodTab.append(logLikehood(ni, y))
        ui = sigm(ni)
        si = ui * (1.0 - ui)
        zi = ni + ((y - ui) / si)
        S = numpy.diag(si)
        mIdit = numpy.identity(x.shape[1])
        w = numpy.linalg.inv((x.transpose().dot(S).dot(x)) + (mIdit.dot(lam))).dot(x.transpose().dot(S).dot(zi))
        normTab.append(numpy.linalg.norm(w - wLast))
        accuracyTab.append(accuracyCalc(ui, y))
        i += 1
        if numpy.linalg.norm(w - wLast) < convStop:
            break
        wLast = w;
    return i, w, ui, logLikelihoodTab, normTab, accuracyTab


# IRLS test
def irlsPredict(x, w):
    print('1')
    resP = sigm(x.dot(w))
    print('2')
    return resP


# Parser dataset
print("Start program ...")
xTrain, yTrain = parser(datasetPath[0])
xTest, yTest = parser(datasetPath[1])
print("File parser")

# Parser IRLS
print("IRLS run start")
lamb = [0.01, 0.5, 0.1, 1, 10, 100]
print("Trainning ...")
dataTab = []
iTi = 0
while iTi < len(lamb):
    i, w, ui, logLikelihoodTab, normTab, accuracyTab = irls(xTrain, yTrain, lamb[iTi])
    dataTabL = [i, ui, w, logLikelihoodTab, normTab, accuracyTab]
    dataTab.append(dataTabL)
    printRes(ui, yTrain, i, logLikelihoodTab, lamb[iTi], normTab, accuracyTab)
    plotDataTrain(i, logLikelihoodTab, normTab, accuracyTab)
    logLikelihoodTab[:] = []
    normTab[:] = []
    accuracyTab[:] = []
    iTi += 1
print("||||||||||||||||||||||||||||||||||||||||||||||")
print("Testing ...")
iTi = 0
lambTab = []
accuracyTab = []
iTab = []
while iTi < len(lamb):
    dataT = dataTab[iTi]
    r = irlsPredict(xTest, dataT[2])
    accuracy = accuracyCalc(r, yTest)
    printRes(r, yTest, 1, dataT[3], lamb[iTi], dataT[4], accuracy)
    lambTab.append(str(lamb[iTi]))
    accuracyTab.append(accuracy)
    iTab.append(dataT[0] + 1)
    iTi += 1
plotDataTest(lambTab, accuracyTab, iTab)
print("IRLS run done")
