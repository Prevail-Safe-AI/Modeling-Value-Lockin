---
license: odc-by
size_categories:
- 1M<n<10M
task_categories:
- text-generation
- question-answering
- text2text-generation
pretty_name: WildChat-1M
dataset_info:
  features:
  - name: conversation_hash
    dtype: string
  - name: model
    dtype: string
  - name: timestamp
    dtype: timestamp[us, tz=UTC]
  - name: conversation
    list:
    - name: content
      dtype: string
    - name: country
      dtype: string
    - name: hashed_ip
      dtype: string
    - name: header
      struct:
      - name: accept-language
        dtype: string
      - name: user-agent
        dtype: string
    - name: language
      dtype: string
    - name: redacted
      dtype: bool
    - name: role
      dtype: string
    - name: state
      dtype: string
    - name: timestamp
      dtype: timestamp[us, tz=UTC]
    - name: toxic
      dtype: bool
    - name: turn_identifier
      dtype: int64
  - name: turn
    dtype: int64
  - name: language
    dtype: string
  - name: openai_moderation
    list:
    - name: categories
      struct:
      - name: harassment
        dtype: bool
      - name: harassment/threatening
        dtype: bool
      - name: harassment_threatening
        dtype: bool
      - name: hate
        dtype: bool
      - name: hate/threatening
        dtype: bool
      - name: hate_threatening
        dtype: bool
      - name: self-harm
        dtype: bool
      - name: self-harm/instructions
        dtype: bool
      - name: self-harm/intent
        dtype: bool
      - name: self_harm
        dtype: bool
      - name: self_harm_instructions
        dtype: bool
      - name: self_harm_intent
        dtype: bool
      - name: sexual
        dtype: bool
      - name: sexual/minors
        dtype: bool
      - name: sexual_minors
        dtype: bool
      - name: violence
        dtype: bool
      - name: violence/graphic
        dtype: bool
      - name: violence_graphic
        dtype: bool
    - name: category_scores
      struct:
      - name: harassment
        dtype: float64
      - name: harassment/threatening
        dtype: float64
      - name: harassment_threatening
        dtype: float64
      - name: hate
        dtype: float64
      - name: hate/threatening
        dtype: float64
      - name: hate_threatening
        dtype: float64
      - name: self-harm
        dtype: float64
      - name: self-harm/instructions
        dtype: float64
      - name: self-harm/intent
        dtype: float64
      - name: self_harm
        dtype: float64
      - name: self_harm_instructions
        dtype: float64
      - name: self_harm_intent
        dtype: float64
      - name: sexual
        dtype: float64
      - name: sexual/minors
        dtype: float64
      - name: sexual_minors
        dtype: float64
      - name: violence
        dtype: float64
      - name: violence/graphic
        dtype: float64
      - name: violence_graphic
        dtype: float64
    - name: flagged
      dtype: bool
  - name: detoxify_moderation
    list:
    - name: identity_attack
      dtype: float64
    - name: insult
      dtype: float64
    - name: obscene
      dtype: float64
    - name: severe_toxicity
      dtype: float64
    - name: sexual_explicit
      dtype: float64
    - name: threat
      dtype: float64
    - name: toxicity
      dtype: float64
  - name: toxic
    dtype: bool
  - name: redacted
    dtype: bool
  - name: state
    dtype: string
  - name: country
    dtype: string
  - name: hashed_ip
    dtype: string
  - name: header
    struct:
    - name: accept-language
      dtype: string
    - name: user-agent
      dtype: string
  splits:
  - name: train
    num_bytes: 6844366367.030628
    num_examples: 837989
  download_size: 3360836020
  dataset_size: 6844366367.030628
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
tags:
- instruction-finetuning
---
# Dataset Card for WildChat

## Dataset Description
 
- **Paper:** https://arxiv.org/abs/2405.01470

- **Interactive Search Tool:** https://wildvisualizer.com ([paper](https://arxiv.org/abs/2409.03753))

- **License:** [ODC-BY](https://opendatacommons.org/licenses/by/1-0/)

- **Language(s) (NLP):** multi-lingual

- **Point of Contact:** [Yuntian Deng](https://yuntiandeng.com/)

### Dataset Summary

WildChat is a collection of 1 million conversations between human users and ChatGPT, alongside demographic data, including state, country, hashed IP addresses, and request headers. We collected WildChat by offering online users free access to OpenAI's GPT-3.5 and GPT-4. In this version, 25.53% of the conversations come from the GPT-4 chatbot, while the rest come from the GPT-3.5 chatbot. The dataset contains a broad spectrum of user-chatbot interactions that are not previously covered by other instruction fine-tuning datasets: for example, interactions include ambiguous user requests, code-switching, topic-switching, political discussions, etc. WildChat can serve both as a dataset for instructional fine-tuning and as a valuable resource for studying user behaviors. Note that this version of the dataset only contains non-toxic user inputs/ChatGPT responses.

### Updates

**2024-10-17: Content Update.** Conversations flagged by [Niloofar Mireshghallah](https://homes.cs.washington.edu/~niloofar/) and her collaborators in ["Breaking News: Case Studies of Generative AI's Use in Journalism"](https://arxiv.org/abs/2406.13706) for containing PII or sensitive information have been removed from this version of the dataset.

**2024-07-22: Content Update.** All toxic conversations identified by the OpenAI Moderations API or Detoxify have been removed from this version of the dataset.

**2024-06-26: License Change.** We have updated the license of WildChat to [ODC-BY](https://opendatacommons.org/licenses/by/1-0/). This change is retroactively applied to any previous downloads under the ImpACT license.

### Full Version with Toxic Content

For access to the full version of the WildChat dataset, which includes toxic conversations flagged by the OpenAI Moderations API or Detoxify, please refer to [WildChat-1M-Full](https://huggingface.co/datasets/allenai/WildChat-1M-Full). This version requires approval and justification for why toxic data is needed.

### Languages

68 languages were detected in WildChat.

### Personal and Sensitive Information

The data has been de-identified with Microsoft Presidio and hand-written rules by the authors.

### Data Fields

- `conversation_hash` (string): The hash of each conversation's content. This is not a unique key, as different conversations with the same content will share the same hash. For unique identifiers, use `turn_identifier` within each turn.
- `model` (string): The underlying OpenAI model, such as gpt-3.5-turbo or gpt-4.
- `timestamp` (timestamp): The timestamp of the last turn in the conversation in UTC.
- `conversation` (list): A list of user/assistant utterances. Each utterance is a dictionary containing the `role` of the speaker (user or assistant), the `content` of the utterance, the detected `language` of the utterance, whether the content of the utterance is considered `toxic`, and whether PII has been detected and anonymized (`redacted`). For user turns, there's also the hashed IP address `hashed_ip` of the turn, the state `state` and country `country` inferred from the original IP address, and the request headers `header` (which might be useful for linking multiple conversations from the same user when used in conjunction with `hashed_ip`). For assistant turns, there's a field `timestamp` which is the time when the backend server receives the full response from ChatGPT. For both user and assistant turns, there's a unique idenifier `turn_identifier`.
- `turn` (int): The number of turns in the conversation. A turn refers to one round of user-assistant interaction.
- `language` (string): The language of the conversation. Note that this is the most frequently detected language in the utterances of the conversation.
- `openai_moderation` (list): A list of OpenAI Moderation results. Each element in the list corresponds to one utterance in the conversation. When the content of an utterance is an empty string, the corresponding moderation reult is set to be an empty dictionary.
- `detoxify_moderation` (list): A list of Detoxify results. Each element in the list corresponds to one utterance in the conversation. When the content of an utterance is an empty string, the corresponding Detoxify reult is set to be an empty dictionary.
- `toxic` (bool): Whether this conversation contains any utterances considered to be toxic by either OpenAI Moderation or Detoxify.
- `redacted` (bool): Whether this conversation contains any utterances in which PII is detected and anonymized.
- `state` (string): The state inferred from the most common IP address in the conversation. Its value is sometimes `None` when GeoIP2 does not identify the state of an IP address.
- `country` (string): The country inferred from the most common IP address in the conversation. Its value is sometimes `None` when GeoIP2 does not identify the country of an IP address.
- `hashed_ip` (string): The most common hashed IP address in the conversation.
- `header` (string): The request header containing information about operating system, browser versions, and accepted languages. This field might be useful for linking multiple conversations from the same user when used in conjunction with `hashed_ip`. Note that every turn in a conversation has the same header, as this is the way we linked turns into conversations.

### Empty User Inputs

This dataset includes a small subset of conversations where users submitted empty inputs, sometimes leading to hallucinated responses from the assistant. This issue, first noticed by @yuchenlin, arises from the design of our Huggingface chatbot used for data collection, which did not restrict the submission of empty inputs. As a result, users could submit without entering any text, causing the assistant to generate responses without any user prompts. This occurs in a small fraction of the dataset.

### Licensing Information

WildChat is now made available under the [**ODC-BY License**](https://opendatacommons.org/licenses/by/1-0/). This change is retroactively applied to any previous downloads under the ImpACT license.

### Citation Information

Please consider citing [our paper](https://arxiv.org/abs/2405.01470) if you find this dataset useful:
```
@inproceedings{
  zhao2024wildchat,
  title={WildChat: 1M Chat{GPT} Interaction Logs in the Wild},
  author={Wenting Zhao and Xiang Ren and Jack Hessel and Claire Cardie and Yejin Choi and Yuntian Deng},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=Bl8u7ZRlbM}
}
```

```
@misc{deng2024wildvisopensourcevisualizer,
  title={WildVis: Open Source Visualizer for Million-Scale Chat Logs in the Wild}, 
  author={Yuntian Deng and Wenting Zhao and Jack Hessel and Xiang Ren and Claire Cardie and Yejin Choi},
  year={2024},
  eprint={2409.03753},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2409.03753}, 
}
```