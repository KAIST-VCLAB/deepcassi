# High-Quality Hyperspectral Reconstruction<br>Using a Spectral Prior
This work was presented at Bangkok in SIGGRAPH ASIA 2017. Visit our project [page](http://vclab.kaist.ac.kr/siggraphasia2017p1/index.html).

## Requirements
We developed the code on Ubuntu 14.04, but we believe that our code will run on other versions of Ubuntu, Windows, and OSX as well. When using different versions of Python and Tensorflow, minor modifications, such as function names, in the code will be necessary.

* Python 2.7
* Tensorflow [r0.12](https://www.tensorflow.org/versions/r0.12/)
* numpy
* scipy
* cv2 (opencv for python)

## How to Execute Demo
We have written a demo code in 'demo_HQHS_recon.py'. In the main function of it, there called two different functions: 
* demo_recon_synthetic_CAVE() : this function demonstrates our reconstruction for [CAVE](http://www.cs.columbia.edu/CAVE/databases/multispectral/) dataset
* demo_recon_synthetic_KAIST(): this function demonstrates our reconstruction for our [KAIST](http://vclab.kaist.ac.kr/siggraphasia2017p1/index.html) dataset

In the definition of each demo function, you are allowed to change parameter values, such as rho, sparsity, alpha-fidelity, learning-rate, the number of iterations, and input/output filenames.

Note: 'demo_AE_inference'py' shows how to feed and reconstruct inputs to our autoencoder to generate the corresponding outputs.
