# High-Quality Hyperspectral Reconstruction<br>Using a Spectral Prior
![alt text](./paper-teaser.png "Paper Teasear")
This work was presented at Bangkok in [**SIGGRAPH ASIA 2017**](https://sa2017.siggraph.org/). Visit our project [page](http://vclab.kaist.ac.kr/siggraphasia2017p1/index.html).

## Requirements
We developed the code on **Ubuntu 14.04**, but we believe that our code will run on other versions of Ubuntu, Windows, and OSX as well. When using different versions of Python and Tensorflow, minor modifications, such as function names, in the code will be necessary.

* Python 2.7
* Tensorflow [r0.12](https://www.tensorflow.org/versions/r0.12/)
* numpy
* scipy
* cv2 (opencv for python)

## How to Execute Demo
We have written a demo code in **demo_HQHS_recon.py**. In the main function of it, there called two different functions: 
* **demo_recon_synthetic_CAVE()** : this function demonstrates our reconstruction for [**CAVE**](http://www.cs.columbia.edu/CAVE/databases/multispectral/) dataset
* **demo_recon_synthetic_KAIST()**: this function demonstrates our reconstruction for our [**KAIST**](http://vclab.kaist.ac.kr/siggraphasia2017p1/index.html) dataset

In the definition of each demo function, you are allowed to change parameter values, such as rho, sparsity, alpha-fidelity, learning-rate, the number of iterations, and input/output filenames. You can change the modulation matrix, **Phi**, in **modulation.py**.

#### Note: 'demo_AE_inference.py' shows how to feed and reconstruct inputs to our autoencoder to generate the corresponding outputs.

## Contacts
For questions, please send an email to **[inchangchoi](http://inchangchoi.info)@vclab.kaist.ac.kr**

## Acknowldegments
[Min H. Kim](http://vclab.kaist.ac.kr/minhkim/index.html) acknowledges Korea NRF grants (2016R1A2B2013031, 2013M3A6A6073718) and additional support by Korea Creative Content Agency (KOCCA) in Ministry of Culture, Cross-Ministry Giga KOREA Project (GK17P0200), Sports and Tourism (MCST), Samsung Electronics (SRFC-IT1402-02), and an ICT R&D program of MSIT/IITP of Korea (R7116-16-1035). 

[Diego Gutierrez](http://giga.cps.unizar.es/~diegog/) acknowledges funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (CHAMELEON project, grant agreement No 682080), and from the Spanish Ministerio de Economía y Competitividad (project TIN2016-78753-P).
