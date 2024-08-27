import json
from model_handler.oss_handler import OSSHandler
from model_handler.utils import ast_parse
from model_handler.constant import DEFAULT_SYSTEM_PROMPT
from model_handler.utils import (
    ast_parse,
    system_prompt_pre_processing,
    _get_language_specific_hint,
    func_doc_language_specific_pre_processing,
)


SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
also point it out.
"""

TOOL_PROMPT = """Here is a list of functions in JSON format that you can invoke:\n{functions}.
Should you decide to return the function call(s). \nPut it in the format of [func1(params_name=params_value, params_name2=params_value2...), func2(params)]\n
NO other text MUST be included.
"""

class ToolACEHandler(OSSHandler):
    def __init__(self, model_name, temperature=0.001, top_p=1, max_tokens=1000) -> None:
        super().__init__(model_name, temperature, top_p, max_tokens)

    def _format_prompt(prompts, function, test_category):

        formatted_prompt = "<|begin_of_text|>"

        for prompt in prompts:
            formatted_prompt += f"<|start_header_id|>{prompt['role']}<|end_header_id|>\n\n{prompt['content']}<|eot_id|>"

        formatted_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"

        return formatted_prompt
    
    @staticmethod
    def process_input(
        test_question,
        format_prompt_func,
        use_default_system_prompt,
        include_default_formatting_prompt,
    ):
        prompts = []
        for question in test_question:
            test_category = question["id"].rsplit("_", 1)[0]
            functions = func_doc_language_specific_pre_processing(
                question["function"], test_category
            )
            # prompt here is a list of dictionaries, one representing a role and content
            if use_default_system_prompt:
                question["question"] = system_prompt_pre_processing(
                    question["question"], 
                    f"{SYSTEM_PROMPT}\n{TOOL_PROMPT.format(functions=functions)}"
                )

            language_hint = _get_language_specific_hint(test_category)
            question["question"][-1]["content"] += f"\n{language_hint}"


            prompts.append(format_prompt_func(question["question"], functions, test_category))

        return prompts

    def inference(
        self,
        test_question,
        num_gpus,
        gpu_memory_utilization,
        format_prompt_func=_format_prompt,
        max_model_len=4096
    ):
        return super().inference(
            test_question,
            num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            format_prompt_func=format_prompt_func,
            use_default_system_prompt=True,
            include_default_formatting_prompt=True,
            max_model_len=max_model_len
        )

    def decode_ast(self, result, language="Python"):
        func = result
        func = func.replace("\n", "")  # remove new line characters
        if not func.startswith("["):
            func = "[" + func
        if not func.endswith("]"):
            func = func + "]"
        decoded_output = ast_parse(func, language)
        return decoded_output

    def decode_execute(self, result):
        func = result
        func = func.replace("\n", "")  # remove new line characters
        if not func.startswith("["):
            func = "[" + func
        if not func.endswith("]"):
            func = func + "]"
        decode_output = ast_parse(func)
        execution_list = []
        for function_call in decode_output:
            for key, value in function_call.items():
                execution_list.append(
                    f"{key}({','.join([f'{k}={repr(v)}' for k, v in value.items()])})"
                )
        return execution_list
