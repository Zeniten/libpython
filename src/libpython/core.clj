(ns libpython.core
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py]))

;; load Python libraries
(require-python '[transformers :as transformers])
(require-python '[torch :as torch])

(def text "Who was Jim Henson? Jim henson was a")
;; load pre-trained model tokenizer (vocabulary)
(def tokenizer (py/$a transformers/GPT2Tokenizer from_pretrained "gpt2"))
;; encode text input
(def indexed-tokens (py/$a tokenizer encode text))
;; convert indexed-tokens to PyTorch tensor
(def token-tensor (torch/tensor [indexed-tokens]))
