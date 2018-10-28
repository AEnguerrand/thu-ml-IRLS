import os
import numpy

# Training: a9a/a9a / Test: a9a/a9a.t
datasetPath = 'a9a/a9a'
maxIt = 3


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


# IRLS algo
def irls(x, y, lam):
    i = 0
    w = numpy.zeros(x.shape[1])
    w0 = numpy.log(numpy.average(y) / (1 - numpy.average(y)))
    while i < maxIt:
        print("IRLS: IT")
        ni = w0 + x.dot(w)
        ui = sigm(ni)
        si = ui * (1.0 - ui)
        zi = ni + ((y - ui) / si)
        S = numpy.diag(si)
        u = S.dot(zi)
        mIdit = numpy.identity(x.shape[1])
        w = numpy.linalg.inv((x.transpose().dot(S).dot(x)) + (mIdit * lam)).dot(x.transpose().dot(S).dot(zi))
        print(w)
        i += 1
    return i, w, ui


# Parser dataset
print("Start program ...")
x, y = parser(datasetPath)
print("File parser")
# Parser IRLS
i, w, ui = irls(x, y, 1)
print("IRLS run")
# Print
u = 0
while u < len(y):
    print(round(ui[u]), "-", y[u])
    u += 1


