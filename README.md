# High-Quality Hyperspectral Reconstruction<br>Using a Spectral Prior
![teaser](./paper-teaser.png "Paper Teasear")

## General Information
- Codename: DeepCASSI (ACM SIGGRAPH Asia 2017)
- Writers:  Inchang Choi (inchangchoi@vclab.kaist.ac.kr), Daniel S. Jeon (sjjeon@vclab.kaist.ac.kr), Giljoo Nam (gjnam@vclab.kaist.ac.kr), Min H. Kim (minhkim@vclab.kaist.ac.kr)
- Institute: KAIST Visual Computing Laboratory

For information please see the paper:
 - High-Quality Hyperspectral Reconstruction Using a Spectral Prior
   [**ACM SIGGRAPH ASIA 2017**](https://sa2017.siggraph.org/), Inchang Choi, Daniel S. Jeon, Giljoo Nam, Diego Gutierrez, Min H. Kim
Visit our project [http://vclab.kaist.ac.kr/siggraphasia2017p1/](http://vclab.kaist.ac.kr/siggraphasia2017p1/) for the hyperspectral image dataset.

Please cite our paper if you use any of the free material in this website:
- Bibtex:
@Article{DeepCASSI:SIGA:2017,<br>
  author  = {Inchang Choi and Daniel S. Jeon and Giljoo Nam <br>
             and Diego Gutierrez and Min H. Kim},<br>
  title   = {High-Quality Hyperspectral Reconstruction <br>
             Using a Spectral Prior},<br>
  journal = {ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2017)},<br>
  year    = {2017},<br>
  volume  = {36},<br>
  number  = {6},<br>
  pages   = {218:1-13},<br>
  doi     = "10.1145/3130800.3130810",<br>
  url     = "http://dx.doi.org/10.1145/3130800.3130810",<br>
  }

## License Information

- Inchang Choi, Daniel S. Jeon, Giljoo Nam, Min H. Kim have developed this software and related documentation (the "Software"); confidential use in source form of the Software, without modification, is permitted provided that the following conditions are met:
  1. Neither the name of the copyright holder nor the names of any contributors may be used to endorse or promote products derived from the Software without specific prior written permission. 
  2. The use of the software is for Non-Commercial Purposes only. As used in this Agreement, "Non-Commercial Purpose" means for the purpose of education or research in a non-commercial organisation only. "Non-Commercial Purpose" excludes, without limitation, any use of the Software for, as part of, or in any way in connection with a product (including software) or service which is sold, offered for sale, licensed, leased, published, loaned or rented. If you require a license for a use excluded by this agreement, please email [minhkim@kaist.ac.kr].

- License:  GNU General Public License Usage
Alternatively, this file may be used under the terms of the GNU General Public License version 3.0 as published by the Free Software Foundation and appearing in the file LICENSE.GPL included in the packaging of this file. Please review the following information to ensure the GNU General Public License version 3.0 requirements will be met: http://www.gnu.org/copyleft/gpl.html.

- Warranty: KAIST-VCLAB MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. KAIST-VCLAB SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.


## Requirements
We developed the codes on **Ubuntu 14.04** and **Ubuntu 16.04** (which were explicitly tested) but believe that our codes could be executed on other platforms of Ubuntu, Windows, and OSX as well. When using different versions of Python and Tensorflow, minor modifications, such as function names, in the codes would be necessary.
* ~~Python 2.7~~ -> Python 3.6
* ~~Tensorflow [r0.12]~~ -> Tensorflow [1.10.0](https://www.tensorflow.org)
* numpy
* scipy
* cv2 (opencv for python)

## How to Execute Demo
We have written a demo code in `demo_HQHS_recon.py`. In the main function of it, there called two different functions: 
* **demo_recon_synthetic_CAVE()** : this function demonstrates our reconstruction for [**CAVE**](http://www.cs.columbia.edu/CAVE/databases/multispectral/) dataset
* **demo_recon_synthetic_KAIST()**: this function demonstrates our reconstruction for our [**KAIST**](http://vclab.kaist.ac.kr/siggraphasia2017p1/index.html) dataset

In the definition of each demo function, you are allowed to change parameter values, such as rho, sparsity, alpha-fidelity, learning-rate, the number of iterations, and input/output filenames. You can change the modulation matrix, **Phi**, in `modulation.py`. The inputs to our code are '.mat' files of Matlab, which contain two variables: 'img_hs' and 'wvls2b'. 'img_hs' is a 31-channel hyperspectral image in uint16 format whose intensity range is 0 ~ 65535, and 'wvls2b' is an array of double indicating the wavelength of each channel. Refer to 'inputs/synthetic/KAIST/scene01.mat'. 

#### Note: `demo_AE_inference.py` shows how to feed and reconstruct inputs to our autoencoder to generate the corresponding outputs.

## Contacts
For questions, please send an email to **[inchangchoi](http://inchangchoi.info)@vclab.kaist.ac.kr**

