![alt-text](https://github.com/stanislav-chekmenev/debugging-dl-models/blob/master/assets/dave_i_cant.jpg)
### Material for the class on debugging deep learning models at Data Science Retreat.
#### Useful resources for debugging:
- [Debug a deep learning network](https://medium.com/@jonathan_hui/debug-a-deep-learning-network-part-5-1123c20f960d): A nicely written blog post on how to debug a deep learning model.
- [Troubleshooting deep learning models](https://www.youtube.com/watch?v=GwGTwPcG0YM&feature=youtu.be): A great video to watch where many debugging steps are summarized in a comprehensive way.
- [Machine learning yearning](https://www.deeplearning.ai/machine-learning-yearning/): A practical manual written by Andrew Ng, which gives a full overview how one should structure a deep learning project.
- [Dive into deep learning](https://d2l.ai/index.html): An amazing website with theory of deep learning, code examples, exercises. It starts from basics and covers the most advanced topics in DL.
- [Deep learning](https://www.deeplearningbook.org/): The Bible of deep learning written by Ian Goodfellow and Yoshua Bengio and Aaron Courville. If you wanna go **deep**, this book is a must to read.
 
#### Some other resources, explaining notebook materials:
- [Understanding LSTM networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/): A clear written post on LSTMs. Good for a quick overview and recalling some basics.
- [Batch normalization explained](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c): A blog post explaining batch normalization in detail.
- [See-RNN package](https://github.com/OverLordGoldDragon/see-rnn): A package that helps to visualize RNNs. Check it out if you want to dig into RNNs.
- [Gradient clipping](http://proceedings.mlr.press/v28/pascanu13.html): An article on gradient clipping in RNNs. 
- [Axiomatic attribution for deep networks](https://arxiv.org/abs/1703.01365): An article about Integrated Gradients, which is a useful tool for debugging neural nets.
- [Attribution baselines](https://distill.pub/2020/attribution-baselines/): A really well written blog post on importance of choosing a good baseline for Integrated Gradients.

### Prerequisites:

- Clone the repo: 
```bash
git clone https://github.com/stanislav-chekmenev/debugging-dl-models
```

- Please, create a new virtual environment with Python=3.6. Feel free to use any of your choice. 
Conda:
```bash
conda create --name <name> python=3.6.9
conda activate <name>
```
 Virtualenv:
```bash
python3 -m venv <path/to/venv>
source <path/to/venv>/bin/activate
```
- Upgrade pip, it might be of an old version
```bash
pip install pip --upgrade
```

- Install requirements:
```bash
pip -r install requirements.txt
```

- Install PyCharm:
	- Follow [this](https://www.jetbrains.com/help/pycharm/installation-guide.html) manual for the installation details.


That should be sufficient to run everything. Thank you!






