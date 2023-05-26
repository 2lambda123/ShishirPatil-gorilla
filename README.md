# Gorilla: Large Language Model Connected with Massive APIs
By Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez  ([Project Website](https://shishirpatil.github.io/gorilla/))

![](https://github.com/ShishirPatil/gorilla/blob/gh-pages/assets/img/logo.png)

_See the paper!_ [https://arxiv.org/abs/2305.15334](https://arxiv.org/abs/2305.15334)

_Join our Discord!_ [https://discord.gg/3apqwwME](https://discord.gg/3apqwwME) 

`Gorilla` enables LLMs to use tools by invoking APIs. Given a natural language query, Gorilla comes up with the semantically- and syntactically- correct API to invoke. With Gorilla, we are the first to demonstrate how to use LLMs to invoke 1,600+ (and growing) API calls accurately while reducing hallucination. Join us, as we try to build the API store for LLMs! Hop on our Discord, or open a PR, or email us if you would like to have your API incorporated as well.

## Repository Organization

We include the `APIBench` dataset created by self-instruct in `data/apibench`. All the 1640 API documentation is in `data/api`. We convert this into a LLM-friendly chat format for evaluation, and the questions are in `eval/eval-data/questions`, and the corresponding responses are in `eval/eval-data/responses`.  We have also included the evaluation scripts are in `eval/eval-scripts`. This would be entirely sufficient to train Gorilla yourself, and reproduce our results.
Additionally, to make it more accessible, we will also release the model weights soon! Either way, if you run into any issues please feel free to reach out to us either through Discord or email or raise a Github issue.

## Get Started 

### Install Dependencies

Use the following command to install dependencies: 

```bash
conda create -n gorilla python=3.8
conda activate gorilla
pip install -r requirements.txt
```

## Abstract

From our [paper](https://arxiv.org/abs/2305.15334):

```text
Large Language Models (LLMs) have seen an impressive wave of advances recently, with models 
now excelling in a variety of tasks, such as mathematical reasoning and program synthesis. 
However, their potential to effectively use tools via API calls remains unfulfilled. 
This is a challenging task even for today's state-of-the-art LLMs such as GPT-4, largely 
due to their inability to generate accurate input arguments and their tendency to hallucinate 
the wrong usage of an API call. We release Gorilla, a finetuned LLaMA-based model that surpasses 
the performance of GPT-4 on writing API calls. When combined with a document retriever, 
Gorilla demonstrates a strong capability to adapt to test-time document changes, enabling 
flexible user updates or version changes. It also substantially mitigates the issue of 
hallucination, commonly encountered when prompting LLMs directly. To evaluate the model's 
ability, we introduce APIBench, a comprehensive dataset consisting of HuggingFace, TorchHub, 
and TensorHub APIs. The successful integration of the retrieval system with Gorilla demonstrates 
the potential for LLMs to use tools more accurately, keep up with frequently updated documentation, 
and consequently increase the reliability and applicability of their outputs. 
```

## FAQ(s)

Can we use Gorilla with Langchain, Toolformer, AutoGPT etc?

Absolutely! You've highlighted a great aspect of our tools. Gorilla is  an  end-to-end model, specifically tailored to serve correct API calls without requiring any additional coding. It's designed to work as part of a wider ecosystem and can be flexibly integrated with other tools.

Langchain, is a versatile developer tool. Its "agents" can efficiently swap in any LLM, Gorilla included, making it a highly adaptable solution for various needs.

AutoGPT, on the other hand, concentrates on the art of prompting GPT series models. It's worth noting that Gorilla, as a fully fine-tuned model, consistently shows remarkable accuracy, and lowers hallucination, outperforming GPT-4 in making specific API calls.

Now, when it comes to ToolFormer, Toolformer zeroes in on a select set of tools, providing specialized functionalities. Gorilla, in contrast, has the capacity to manage thousands of API calls, offering a broader coverage over a more extensive range of tools.

The beauty of these tools truly shines when they collaborate, complementing each other's strengths and capabilities to create an even more powerful and comprehensive solution. This is where your contribution can make a difference. We enthusiastically welcome any inputs to further refine and enhance these tools. 

## Citation
```text
@article{patil2023gorilla,
  title={Gorilla: Large Language Model Connected with Massive APIs},
  author={Shishir G. Patil and Tianjun Zhang and Xin Wang and Joseph E. Gonzalez},
  year={2023},
  journal={arXiv preprint arXiv:2305.15334},
} 
```




