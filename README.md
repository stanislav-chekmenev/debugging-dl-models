![alt-text](https://github.com/stanislav-chekmenev/debugging-dl-models/blob/master/assets/dave_i_cant.jpg)
### Material for the class on debugging deep learning models at Data Science Retreat.
#### Useful resources for debugging:
- [Debug a deep learning network](https://medium.com/@jonathan_hui/debug-a-deep-learning-network-part-5-1123c20f960d): A nicely written blog post on how to debug a deep learning model.
- [Recipe for training neural networks](http://karpathy.github.io/2019/04/25/recipe/): An Andrej Karpathy's blog post on how to train neural nets.
- [Troubleshooting deep learning models](https://www.youtube.com/watch?v=GwGTwPcG0YM&feature=youtu.be): A great video to watch where many debugging steps are summarized in a comprehensive way.
- [Machine learning yearning](https://www.deeplearning.ai/machine-learning-yearning/): A practical manual written by Andrew Ng, which gives a full overview how one should structure a deep learning project.
- [Bayesian optimization](http://krasserm.github.io/2018/03/21/bayesian-optimization/): Bayesian optimization for hyperparameter search.
- [Dive into deep learning](https://d2l.ai/index.html): An amazing website with theory of deep learning, code examples, exercises. It starts from basics and covers the most advanced topics in DL.
- [Deep learning](https://www.deeplearningbook.org/): The Bible of deep learning written by Ian Goodfellow and Yoshua Bengio and Aaron Courville. If you wanna go **deep**, this book is a must to read.
 
#### Some other resources, explaining notebook materials:
- [Understanding LSTM networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/): A clear written post on LSTMs. Good for a quick overview and recalling some basics.
- [Visualizing What Batch Normalization Is and Its Advantages](https://towardsdatascience.com/visualizing-what-batch-normalization-is-and-its-advantages-a49bbcd2fd86): A blog post explaining batch normalization in detail.
- [Gradient clipping](http://proceedings.mlr.press/v28/pascanu13.html): An article on gradient clipping in RNNs. 
- [Teaching Large Language Models to Self-Debug](https://arxiv.org/abs/2304.05128): A paper about Rubber Duck self-debugging with LLMs.
- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction.html): LangChain documentation page.

### Prerequisites:

- Clone the repo: 
```bash
git clone https://github.com/stanislav-chekmenev/debugging-dl-models
```

- Install VS Code:
	- Follow [this](https://code.visualstudio.com/docs/setup/setup-overview) link for the installation details.
	- If you are on Linux Ubuntu starting from 16.04, then please use the following command:
	```bash
	sudo snap install --classic code
	```

- Create a new project and a virtual environment for it:
	- Start VS Code and choose the `project` directory as the root directory for the VS code project.
	- Create and activate a new virtual environment with **python 3.12**. Feel free to use any virtual environment of your choice. I prefer `python3-venv` for its simplicity. It also works good with Linux:

		Conda:
		```bash
		conda create --name <name> python=3.12
		conda activate <name>
		```

		Virtual environment. For Ubuntu 24.04:
		```bash
		python3 --version # make sure you have python 3.12 installed
		sudo apt install python3-venv
		python3 -m venv <path/to/venv>
		source <path/to/venv>/bin/activate
		```

- Upgrade pip, it might be of an old version
```bash
pip install pip --upgrade
```

- Install requirements:
```bash
pip install -r requirements.txt
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

### Class material:

- [Colab 0: Introduction to debugging deep learning models](https://colab.research.google.com/drive/1uZjPUJkmvbqNHJ1x0e1wnkpxWLhpInwQ)
- [Colab 1: Implementation recipe of a bug free model](https://colab.research.google.com/drive/1GUZVMKLEr2fC4gzsExxsv_V8eIlB7JVp)
- [Colab 2: Most common bugs](https://colab.research.google.com/drive/1Gk62sP-k_FyWlOXXf8wW9X8ONWxd3Vsw)
- [Colab 3: Rubber duck debugging with LLMs](https://colab.research.google.com/drive/1p_vf1mkhU6JZnMdrx1WYUYBg08RJ__cV?usp=drive_link)
