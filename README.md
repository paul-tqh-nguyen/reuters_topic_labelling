# Reuters Newswire Multi-label Topic Classification

We used deep-learning methods to perform multi-label text classification on the [Reuters-21578 Text Categorization Collection](http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html).

The architectures explored and their performances are shown below:

| Architecture | F1 Score |
| ----------- | ----------- |
| CNN | 0.7655879 |
| Dense | 0.6601627 |
| RNN (LSTM+Attention) | 0.1065762 |

A full report of our findings can be found at https://paul-tqh-nguyen.github.io/reuters_topic_labelling/.

The tools utilized in our exploration include:
- [PyTorch](https://pytorch.org/)
- [TorchText](https://pytorch.org/text/)
- [NLTK](https://www.nltk.org/)
- [bs4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Pandas](https://pandas.pydata.org/)
- Many more (see our [environment.yml](https://github.com/paul-tqh-nguyen/reuters_topic_labelling/blob/master/environment.yml) if you're having trouble getting the functionality running or [contact me](https://paul-tqh-nguyen.github.io/about/)).