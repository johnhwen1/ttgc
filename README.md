# Twisted Torus Model of Grid Cells
This repo contains code to instantiate a twisted torus model of grid cell activity. This model is based on work by Guanella et al. 2007 [[1]](#1) but includes two additions: sources of noise and landmark input. Model details are included below.

## Getting started
You can try out the model without having to install anything by running this Google Colab <a target="_blank" href="https://colab.research.google.com/github/https://colab.research.google.com/github/johnhwen1/ttgc/blob/main/examples/ttgc.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>. Alternatively, you can install the model locally and run the same example notebook (see below for instructions).

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
### Activity and Stabilization
A set of $N$ comodular grid cells's activity is modeled. The activity of neuron $i$ at time $t$ is given by the following:

<p align="center">
$A_i(t) = B_i(t) + \tau\bigg(\frac{B_i(t)} {{< B_j(t-1) >}_{j=1}^{N}} - B_i(t)\bigg),$
</p>

where $\tau$ represents a stabilization factor, ${< \space .\space>}_{j=1}^{N}$ is the mean over cells in the network, and $B_i(t)$ is a linear transfer function defined as follows:

<p align="center">
$B_i(t) = A_i(t-1) + \sum_{j=1}^{N}A_j(t-1)w_{ji}(t-1)$
</p>

where $w_{ji}(t-1)$ is the weight from cell $j$ to cell $i$ at time $t-1$, with $i,j \in \lbrace 1, 2, ..., N\rbrace$.

Neurons are initialized with random activity uniformly between $0$ and $1/\sqrt N$

### Attractor Dynamics
When the agent is stationary, the weight between neuron $i$ and $j$ is defined as follows:

<p align="center">
$w_{ij} = I \exp \bigg(- \frac{\|c_i - c_j\|^2_{tri}} {\sigma^2}\bigg) - T$
</p>

The weight is dependent on the relative "positions" of cells $i$ and $j$. The position of neuron $i$ is defined as ${c_i} = (c_{i_{x}}\space ,\space c_{i_{y}}),$ where $c_{i_{x}} = (i_x− 0.5)/N_x,$ and $c_{i_{y}} = \frac{\sqrt3}{2} (i_y− 0.5)/N_y$ with $i_x \in \lbrace1, 2, ..., N_x\rbrace$ and $i_y \in \lbrace1, 2, ..., N_y\rbrace$, and where $N_x$ and $N_y$ are the number of columns and rows in the cells matrix and $i_x$ and $i_y$ the column and the row numbers of cell $i$. 

Additionally, global parameters that govern the relationship between all pairs of cells include $I$, the intensity parameter, $\sigma$ the size of the Gaussian, $T$ the shift parameter (see the referenced paper for more details).

Finally, the key to getting triangular grid instead of square ones is to use a distance metric defined as follows: 
<p align="center">
$\text{dist}_{tri}(c_i, c_j)$ := $\| c_i - c_j\|_{tri} = \text{min}_{k=1}^7 \| c_i − \space  c_j +  \space s_k\|,$ 
</p>

where

$s_1 := (0, 0)$

$s_2 := (−0.5, \frac{\sqrt3}{2})$

$s_3 := (−0.5, -\frac{\sqrt3}{2})$

$s_4 := (0.5, \frac{\sqrt3}{2})$

$s_5 := (0.5, -\frac{\sqrt3}{2})$

$s_6 := (−1, 0)$

$s_7 := (1, 0)$

<p align="left">
and where $\|.\|$ is the Euclidean norm.
</p>

### Modulation
When the agent is moving, the weight between neurons $i$ and $j$ becomes modulated by the velocity $v := (v_x, v_y)$. In essence, the synaptic connections of the network shift in the direction of the agent. This modulation is expressed as follows:

<p align="center">
$w_{ij}(t) =  I \exp \bigg(- \frac{\|c_i - c_j+ \alpha R_{\beta}v(t-1)\|^2_{tri}} {\sigma^2}\bigg) - T$
</p>

The scale and orientation of the grid is dictated by the gain factor $\alpha \in \mathbb{R}^+$ and bias $\beta \in [0, π/3]$. The input of the network is thus modulated and biased by the gain and the bias parameters, with $v \longmapsto \alpha R_{\beta}v$ , where $R_{\beta}$ is the rotation matrix of angle $\beta$.

### Modifications
This model is modified in two key ways from the model described in Guanella et al 2007. The first modification allows for added heading direction noise at each timestep, and the second introduces landmark inputs to the grid cell network. Heading direction noise is added as $\beta_{\text{noisy}}(t) = \beta + \sigma_{\beta} r(t)$, where $\beta$ is the unmodified bias, $\sigma_{\beta}$ regulates the extent of noise, and $r(t)$ is drawn from the standard normal distribution, and $\beta_{\text{noisy}}(t)$ is still constrained such that $\beta_{\text{noisy}}(t) \in [0, π/3]$ The rotation matrix is then calculated using $\beta_{\text{noisy}}(t)$.

Landmark inputs are added with the addition of landmark cells and their unidirectional excitatory synaptic connections to grid cells. When landmarks are present, each landmark $L_{i}$ is associated with its own dedicated landmark cell population. A given landmark cell's activity $A_{L_{i_j}}$ is dependent on the agent's proximity to the landmark's position, where $i \in \lbrace1, ..., N_L\rbrace$ and where $N_L$ is the number of landmarks present and $j \in \lbrace1, ..., N_{Ln}\rbrace$ where $N_{Ln}$ is a global parameter setting the number of landmark cells dedicated to any given landmark. The activity of landmark cell $A_{L_{i_j}}$ is defined as follows:
<p align="center">
$A_{L_{i_j}} = \alpha_{L_i} \exp \bigg(- \frac{\|p(t) - p_{L_i}\|^2} {\big(\frac{1}{2} q_{L_i}\big)^2} \bigg)$ if $\|p(t) - p_{L_i}\| \leq q_{L_i}$ and is otherwise set to $0$.
</p>

where the strength of landmark $L_i$ is governed by $\alpha_{L_i} \in \mathbb{R}^+$, $p(t):= (p_x(t), p_y(t))$ is the position of the agent at time $t$, $p_{L_i} := (p_{L_{i_x}}, p_{L_{i_y}})$ is the position of $L_i$, and $q_{L_i} \in \mathbb{R}^+$ represents the lookahead distance at which landmark $L_i$ begins recruiting the activity of its landmark cells. In the most basic form of this model, each landmark cell sends an excitatory connection to only one grid cell and contributes to its activity through a modification to the linear transfer function:

<p align="center">
$B_i(t) = A_i(t-1) + \sum_{j=1}^{N}A_j(t-1)w_{ji}(t-1) + \sum_{k=1}^{N_{L}} A_{L_k}(t-1) w_{ki}$
</p>

where $w_{ki} = 1$ and $\sum_{i=1}^{N} w_{ki} = 1$, allowing each landmark cell to be connected with only one grid cell. 

A model containing a Hebbian plasticity term between landmark cells and grid cells makes the following modifications:
1. The weight between each landmark cell and the $N$ grid cells is initialized randomly with values between $0$ and $1/\sqrt N$
2. Hebbian plasticity allows changes in weights between landmark cells and grid cells as follows:

$w_{ki}(t+1) = w_{ki}(t) + \alpha_{\text{hebb}} (A_i(t) A_{L_k}(t)) - \alpha_{\text{decay}}$

where $\alpha_{\text{hebb}}$ regulates the extent of Hebbian potentiation and $\alpha_{\text{decay}}$ provides a constant decay factor. Weight values are constrained to be between some minimum $W_{L_{\text{min}}}$ and maximum $W_{L_{\text{max}}}$

## References
<a id="1">[1]</a>
Guanella, A., Kiper, D. & Verschure, P. 
A model of grid cells based on a twisted torus topology. 
Int. J. Neural Syst. 17, 231–240 (2007).
