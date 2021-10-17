# 自然语言处理大作业一：n-gram语言模型平滑算法

实现了n-gram语言模型以及Kneser-Ney discounting 算法

## Requirements
   - python3
   - numpy
   - matplotlib
   - pickle

## Results
| n-gram order | d | UNK threshold | PPL on dev | PPL on test |
| :----------: | :---: | :------: | :---------: | :---------: |
| 2       | 0.82     |   1    | 465.80 | 412.57 |
| 3        | 0.84     |   1    | **3.16** | **2.63** |
| 4        | 0.84    |   1    | 17.00 | 14.88 |

## Prepare dataset
- Put the dataset under the same directory like this:
```
|-- README.md
|-- ngram_discounting.py
|-- dev_set.txt
|-- test_set.txt
|-- train_set.txt
```

## Quick Start
- You can just run:
```bash
    python ngram_discounting.py
```

- For customized use, here is an example:
```python
    m = NGramKneserNeyDiscountingModel()
    m.train()

    # Optional
    # To adjust the hyperparameters on dev set
    # m.dev() 

    m.test()
    save_model(m)

    # To visualize the word frequency distribution:
    # m.plot_freq_rank()    
```