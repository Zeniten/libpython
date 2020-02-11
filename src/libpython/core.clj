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
(def tokens-tensor (torch/tensor [indexed-tokens]))
;; load pre-trained model (weights)
(def model (py/$a transformers/GPT2LMHeadModel from_pretrained "gpt2"))

;; Set the model in evaluation mode to deactivate the DropOut modules.
;; IMPORTANT: This is to have reproducible results during evaluation.
(py/$a model eval)

;; predict all tokens
(def predictions (py/with [r (torch/no_grad)]
                          (first (model tokens-tensor))))

;; get the predicted next sub-word
(def predicted-token (let [last-word-predictions (-> predictions first last)
                           arg-max (torch/argmax last-word-predictions)]
                       (py/$a arg-max item)))

;; decode indexed-tokens + predicted-token
(py/$a tokenizer decode (-> (into [] indexed-tokens)
                           (conj predicted-token)))
