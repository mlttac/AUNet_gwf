# Attention U-Net as a surrogate model for groundwater prediction

Code and data accompanying the manuscript titled "Attention U-Net as a surrogate model for groundwater prediction
", authored by Maria Luisa Taccari, Jonathan Nuttall, Xiaohui Chen, He Wang, Bennie Minnema, Peter K. Jimack

 [Attention U-Net as a surrogate model for groundwater prediction](https://www.sciencedirect.com/science/article/pii/S0309170822000458#:~:text=The%20Attention%20U%2DNet%20model,the%20pattern%20of%20its%20distribution.)
 
# Abstract
Numerical simulations of groundwater flow are used to analyze and predict the response of an aquifer system to its change in state by approximating the solution of the fundamental groundwater physical equations. The most used and classical methodologies, such as Finite Difference (FD) and Finite Element (FE) Methods, use iterative solvers which are associated with high computational cost. This study proposes a physics-based convolutional encoder-decoder neural network as a surrogate model to quickly calculate the response of the groundwater system. Holding strong promise in cross-domain mappings, encoder-decoder networks are applicable for learning complex input-output mappings of physical systems. This manuscript presents an Attention U-Net model that attempts to capture the fundamental input-output relations of the groundwater system and generates solutions of hydraulic head in the whole domain given a set of physical parameters and boundary conditions. The model accurately predicts the steady state response of a highly heterogeneous groundwater system given the locations and piezometric head of up to 3 wells as input. The network learns to pay attention only in the relevant parts of the domain and the generated hydraulic head field corresponds to the target samples in great detail. Even relative to coarse finite difference approximations the proposed model is shown to be significantly faster than a comparative state-of-the-art numerical solver, thus providing a base for further development of the presented networks as surrogate models for groundwater prediction.


##  Data-set:
[Training data](https://drive.google.com/file/d/1MjcE6QXvlyjX6lWJpEf0Uf-CWMH9YJet/view?usp=sharing)
