# ShortcutGroup
code for "Counterfactual Contrastive Learning for Robust Text Classification Based on Word-Group Search" submitted to Information Sciences

All the datasets used are listed in /dataset

Once you've cloned the code into your own environment, you'll need to modify config.py first, specifically changing self.root to your own language model path.

After that, you can follow the steps below to train your model.

1. Leave the following code in train.py to train a weak classifier based on the basic language model for keyword search.
```python
trainer = WeakTrainer(weak_cfg)
trainer.train_weak(weak_cfg)
```

2. Subsequently, based on the trained weak classifier, candidate causal words need to be searched.
```python
trainer.get_global_ig_sub_keywords()
trainer.get_global_ig_for_each_sample()
```

3. Search for the most influential word groups based on the candidate causal words and the weak classifier in SearchWG.py, you can modify **MAX_WG_LENGTH**, **BEEM_WIDTH**, and **antony.json** to define your own search results. Here, **antony.json** It's a predefined antonym table from WordNet for counterfactual flipping your word groups.

4. A robust model is trained based on the searched word groups by train.py.
```python
cfg = DebiasConfig()
trainer = DebiasTrainer(cfg)
trainer.train_Debias(cfg)
```

Once you have trained your model, you can test it by:
```python
trainer.test_CD_debias(cfg)
```
