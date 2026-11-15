Unmodified inference sources from deepseek-ai/DeepSeek-V4.1-Flash at
314113f8a7dadb388215396ba5fbee2c610a04d6 (MIT; see LICENSE).

Source: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6/inference

The sibling kernel.py and checkpoint.py provide the AMD reference implementation;
the original TileLang kernel module is not imported.

encoding.py is also unmodified from the same revision, under encoding/encoding.py.
SHA256: 13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2.
Source: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/encoding/encoding.py
The server uses its chat-mode encoder and completion parser, not a Jinja template.
