from g2s import g2s
import sys
import numpy
from PIL import Image
import requests
from io import BytesIO
import time

# Code by (Dr. Mathieu Gravey) https://www.mgravey.com/
starN = 1
endN = 200

imName = 'stone';
if (len(sys.argv) > 2):
    imName = sys.argv[1]

ti = numpy.array(Image.open(BytesIO(requests.get(
    'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/{}.tiff'.format(imName)).content)));
print(ti)
k = 1.2;
if (len(sys.argv) > 2):
    k = float(sys.argv[2])
print(k)

simN = np.ones((endN - starN, 200,200)) * np.arange(starN, endN)[:, None, None];

jobID = g2s('-a', 'qs', '-ti', ti, '-di', simN * numpy.nan, '-dt', [0], '-ni', simN, '-k', k, '-ki',
            numpy.ones((50, 50)), '-j', 1.0001, '-submitOnly');

progress = g2s('-statusOnly', jobID);
while (progress[0] < 98):
    time.sleep(1)
    progress = g2s('-statusOnly', jobID);

sim, index, t = g2s('-waitAndDownload', jobID);

numpy.savez('qsSim_{}_{}.npz'.format(imName, k), sim=sim, indexMap=index, t=t, k=k, ti=ti, starN=starN, endN=endN)
