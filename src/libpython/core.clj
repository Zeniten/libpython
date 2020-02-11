(ns libpython.core
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py]))

(require-python '[transformers :as transformers])
(require-python '[torch :as torch])

(def tokenizer (py/$a transformers/GPT2Tokenizer from_pretrained "gpt2"))
(def text "Who was Jim Henson? Jim henson was a")
(def indexed-tokens (py/$a tokenizer encode text))
(def token-tensor (torch/tensor [indexed-tokens]))
