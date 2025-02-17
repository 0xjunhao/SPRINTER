This is the offcial implementation of the paper: Speeding up Speculative Decoding via Approximate Verification.

We propose SPRINTER, which utilizes a low-complexity verifier trained to predict if tokens generated from a draft LLM would be accepted by the target LLM. By performing approximate sequential verification, SPRINTER does not require verification by the target LLM and is only invoked when a token is deemed unacceptable. This leads to reducing the number of calls to the larger LLM and can achieve further speedups.

Implementation Steps:

1. Use dataset.py to prepare the dataset for training a verifier (need to login huggingface using user sepecific token)
2. Train a verifier using train.py
3. Run SPRINTER.py to implement speculative decoding via approximate verification.
4. Evaluate the quality of SPRINTER using win_rate_eval.py and using rouge_eval.py