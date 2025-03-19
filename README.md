# Twisted Torus Model of Grid Cells
This repo contains code to instantiate a twisted torus model of grid cell activity. This model is based on work by Guanella et al. 2007 [[1]](#1) but includes two additions: sources of noise and landmark input. Model details are included below.

## Getting started
You can try out the model without having to install anything by running this ![Open in Colab](https://colab.research.google.com/github/johnhwen1/ttgc/blob/main/examples/ttgc.ipynb). Alternatively, you can install the model locally and run the same example notebook (see below for instructions).

### Installing and running locally
Use either of two methods below to save and run the code locally. 
#### Method 1: Pip install
Pip install by running
```
pip install ttgc
```
Then, you can download and run ttgc.ipynb in the [examples](https://github.com/johnhwen1/ttgc/examples) folder.

#### Method 2: Git clone (or download)
Git clone or download this repo, cd into the downloaded repo, and then run

```
pip install .
pip install -r requirements.txt
```
Then, navigate to the examples folder and run the ttgc.ipynb 

## Model details

## References
<a id="1">[1]</a>
Guanella, A., Kiper, D. & Verschure, P. 
A model of grid cells based on a twisted torus topology. 
Int. J. Neural Syst. 17, 231–240 (2007).
