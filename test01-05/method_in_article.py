from llama_index.llms.openai_like import OpenAILike
from typing import Optional, List, Dict
import json
import httpx
import time
import subprocess
import os
import fitz
import faiss
from pathlib import Path
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.node_parser import SentenceSplitter
import requests
from openai import OpenAI
import psutil
import ast
import re
import pandas as pd
from docx import Document


def clean_text(text):
    import re
    text = re.sub(r'[^\S\r\n]+', ' ', text)  # 统一空白字符
    text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\r', '\t'])
    return text.strip()


class AliyunTextEmbedding:
    def __init__(self, api_key: str, api_base: str, model: str):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model

    def _get_embedding(self, text: str) -> List[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        # print(self.model)
        payload = {
            "model": self.model,
            "input": "test"
        }
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
        completion = client.embeddings.create(
            model = self.model,
            input = text,
            dimensions=1024,
            encoding_format="float"
        )
        # print(type(completion))
        result = completion.model_dump_json()
        json_result = json.loads(result)
        return json_result['data'][0]['embedding']

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)


def extract_error_between_markers(text):
    start_marker = "set lines(information) [list ]"
    end_marker = "return [array get lines]"
    lines = text.splitlines()

    extracting = False
    extracted = []

    for line in lines:
        if start_marker in line:
            extracting = True
            continue
        if end_marker in line:
            break
        if extracting:
            extracted.append(line)

    return "\n".join(extracted)


class aadlAgent():
    def __init__(self):
        self.model: Optional[str] = None
        self.key: Optional[str] = None
        self.url: Optional[str] = None
        self.inspectorP: Optional[str] = None
        self.storage: Optional[str] = None
        self.embedding_model: Optional[str] = None
        self.max_attempt = 3
        self.temp_req: Optional[str] = None

        my_client = httpx.Client(
            timeout=httpx.Timeout(60.0, read=180.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        with open("../config.json", "r", encoding="utf-8")as config_file:
            data = json.load(config_file)
            self.url = data.get("aliyun_url")
            self.key = data.get("aliyun_key")
            # self.model = data.get("aliyun_model")
            self.model = data.get("qwen_max_model")
            self.inspector = data.get("AADL_INSPECTOR_PATH")
            self.storage = data.get("storage")
            self.embedding_model = data.get("aliyun_embedding_model")

        self.llm = OpenAILike(
            model=self.model,
            api_base=self.url,
            api_key=self.key,
            context_window=96000,
            is_chat_model=True,
            is_function_calling_model=False,
            timeout=120,
            client=my_client,
        )

        self.embed_llm = LangchainEmbedding(
            AliyunTextEmbedding(api_key=self.key, api_base=self.url, model=self.embedding_model)
        )
        Settings.llm = self.llm
        Settings.embed_model = self.embed_llm

        self.index = None
        self.query_engine = None
        self.load_index()

    def load_pdf(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def build_index_from_pdf(self, pdf_path: str, doc_id: Optional[str] = None):
        raw_text = self.load_pdf(pdf_path)

        parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        chunks = parser.split_text(raw_text)

        documents = [
            Document(text=chunk, metadata={"source": os.path.basename(pdf_path), "doc_id": doc_id or "default"})
            for chunk in chunks
        ]

        dimension = 1024

        print("正在创建新的 FAISS 索引...")
        faiss_index = faiss.IndexFlatL2(dimension)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        if os.access(self.storage, os.W_OK):
            print("可写入")
            try:
                self.index.storage_context.persist()
            except Exception as e:
                print("创建本地向量库异常")
                print(e)
                import traceback
                traceback.print_exc()
        else:
            print("不可写入")

    def load_index(self):
        try:
            vector_store = FaissVectorStore.from_persist_dir("../storage")
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir="../storage"
            )
            self.index = load_index_from_storage(storage_context=storage_context)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("未找到已存在的索引。需要创建新的索引。")
            self.index = None

    def query(self, question: str):
        self.query_engine = self.index.as_query_engine()

        if not self.query_engine:
            raise ValueError("索引未建立，请先调用 build_index_from_pdf 加载文档。")

        response = self.query_engine.query(question)
        # print(type(response))
        return str(response)

    def list_loaded_docs(self) -> List[Dict]:
        if not self.index:
            return []

        docs = []
        for ref_doc_info in self.index.ref_doc_info.values():
            docs.append({
                "file_name": ref_doc_info.metadata.get("source"),
                "doc_id": ref_doc_info.metadata.get("doc_id"),
                "length": len(ref_doc_info.text),
            })
        return docs
    
    def analyse_req_details(self, base_path: str, req_slice: str) -> str:
        """
        需求条目细化
        """
        analysis_prompt = f"""
<background>
你是一名AADL建模助手，任务是分析以文件形式输入的自然语言需求(完整项目的一部分)，识别其中涉及的AADL语言元素，思考需求的细化问题
</background>

<adaptive_thinking_framework>
# 细化需求满足以下要求：
- 建立各个组件与功能的对应关系
- 仅在必要情况下才进行细化，禁止过度拆分
</adaptive_thinking_framework>

<instruction>
# 需求细化的指导策略：
- 只有‘所含组件’和‘内部子组件’中的组件属于该模块，其他组件并不在你细化的范围
- 有些组件存在层叠关系（例如process和thread），你需要思考不同功能应该分属于哪一层次
</instruction>

<output_example>
以下是json格式的输出样例
[
    {{
        "组件": "Acceleration_Process",
        "对应功能": "完成加速度数据采集与预处理;执行格式转换、归一化、滤波操作;保证4ms预处理时限",
        "子组件": "Accelero_Thread",
        "关联关系": [Processor: CortexM4.IMPL]
    }},
    ...
]
</output_example>

<input>
以下是你将阅读的需求模块条目：
{req_slice}
</input>

<output>
你将按照以下要求输出：
- 按照你细化结果，参照output_example格式输出
- ‘组件’: 输入条目中需要定义的组件
- ‘对应功能’: 该组件的功能或性能要求
- ‘子组件’：组件内部直接子组件
- ‘关联关系’：与其他组件的联系
- *禁止*输出其他说明与无关内容，不要在前后给出```json等标识
- *严格遵守*我给出的json结构，以保证我可以正则处理
</output>
"""
        response = self.llm.complete(analysis_prompt)
        match = re.search(r"模块名称：\s*([\w_]+)", req_slice)
        module_name = match.group(1)
        print(str(response))
        response_into_file = json.loads(str(response))
        output_file = f"./{base_path}/step1-analyse_requirement/{module_name}_details.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(response_into_file, f, indent=4, ensure_ascii=False)
        return str(response)

    def analyse_rules(self, base_path: str, req_slice: str) -> str:
        """
        构造相似度匹配前体
        """
        components_match = re.search(r"所含组件：\s*((?:\s{4}.+\n)+)", req_slice)
        component_content = components_match.group(1)
        subcomponents_match = re.search(r"内部子组件：\s*((?:\s{4}.+\n)+)", req_slice)
        subcomponents_content = subcomponents_match.group(1)
        behavior_match = re.search(r"具体行为：\s*((?:\s{4}.+\n)+)", req_slice)
        behavior_content = behavior_match.group(1)
        name_match = re.search(r"模块名称：\s*([\w_]+)", req_slice)
        name_content = name_match.group(1)

        analyse_prompt = f"""
<background>
你是一名AADL建模助手，任务是分析以文件形式输入的自然语言需求(完整项目的一部分)，识别其中涉及的AADL语言元素，思考需求知识扩展问题
</background>

<adaptive_thinking_framework>
# 知识扩展满足以下要求
- 思考当前需求条目对应的aadl概念，在知识库中的表现形式，作为可能的语法提示，以帮助我进行RAG的准备工作
- 至多给出三组最核心的语法提示，使用列表组织结果
</adaptive_thinking_framework>

<output_example>
# 以下是输出样例的列表格式：
["thread implementation DataFusion_Thread.impl\n  properties\n    Compute_Execution_Time => 10 ms .. 15 ms;\n    Deadline => 20 ms;", "memory implementation Flash.IMPL\nproperties\n  Storage_Size => 1 GB;\n  Non_Volatile => true;"]
</output_example>

<input>
# 以下是项目主要包含的内容：
所含组件：{component_content}; 内部子组件：{subcomponents_content}; 具体行为：{behavior_content}
</input>

<output>
# 你将按照以下要求输出结果：
- 你将输出一个列表，其中至多包含三项，每项的内容是你根据input分析的aadl对应概念，在知识库中的表现形式
- *禁止*输出其他无关内容，便于我进行正则
- *严格遵守*列表结构
</output>
"""
        response = self.llm.complete(analyse_prompt)
        with open(f"./{base_path}/step1-analyse_requirement/{name_content}_likable.txt", "w", encoding="utf-8") as f:
            f.write(str(response))
        return str(response)

    def get_rules_reference(self, base_path:str, req_slice: str) -> List | None:
        """
        获取RAG检索结果
        """
        name_match = re.search(r"模块名称：\s*([\w_]+)", req_slice)
        name_content = name_match.group(1)
        example_list  = []
        with open(f"./{base_path}/step1-analyse_requirement/{name_content}_likable.txt", "r", encoding="utf-8") as f:
            data = f.read()
            # for example in example_json:
            # if (result := example_json.get("语法提示")):
            rag_result = self.query(data)
            example_list.append(rag_result)
        if example_list:
            with open(f"./{base_path}/step2-get_rules_by_rag/{name_content}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(example_list))
        return example_list
 
    def build_aadl(self, base_path: str, req_slice: str) -> str:
        """
        创建aadl代码
        """
        name_match = re.search(r"模块名称：\s*([\w_]+)", req_slice)
        name_content = name_match.group(1)
        with open(f"./{base_path}/step1-analyse_requirement/{name_content}_details.json", "r", encoding="utf-8") as f:
            analysis_data = json.load(f)
        with open(f"./{base_path}/step2-get_rules_by_rag/{name_content}.txt", "r", encoding="utf-8") as f:
            rag_result = f.read()
        with open(f"./{base_path}/summary.txt", "r", encoding="utf-8") as f:
            summary = f.read()
        rag_prompt = ""
        if rag_result:
            rag_prompt = f"""

<rag_data>
# 检索的规则遵循以下规则：
- 这些规则来自AS5506C语法库，与项目需求无关，只是语法提示
- 提示可能没有用处，这是你可以忽略掉它
- 以下是规则检索结果:{rag_result}
</rag_data>
"""
        
        prompt = f"""
<background>
你是一名熟悉 AADL（Architecture Analysis and Design Language）语言标准的系统建模专家。请根据以下信息，生成符合AADL语法的结构化代码
</background>

<adaptive_thinking_framework>
# 你将获得以下信息：原始需求(来自一个完整项目的部分)，对需求的分析，项目组件架构，通过知识库检索到的规则(这一项可能不存在)
- *原始需求*是你关注的重点
- *对需求的分析*用来辅助你思考，但它可能有误区，若有冲突依据*原始需求*为准
- *项目组件架构*描述了完整的项目中所含的主要组件和架构关系，能帮助你理解需求调用与连接关系
- *通过知识库检索到的规则*只是提示一些语法规则，它与需求本身无关，提示的规则也不一定是你需要的，可以选择性忽略，有时其本身也不会存在
</adaptive_thinking_framework>

<original_requirement>
# 原始需求的使用遵循以下规则：
- 原始需求描述的内容是一个完整项目的部分功能模块
- *所含组件*与*内部子组件*是你在代码中需要定义或实例化的，而*关联组件*在其他文件有详细定义，它的用途是帮助你进行组件链接
- 为了确保代码能进行检验，仅当需要调用*关联组件*时，你可以简单定义，其他情况下则不需要编写定义
- *功能说明*和*具体行为*描述代码的功能用途，你需要通过aadl代码实现
- 仅当*功能说明*和*具体行为*某些条目无法用aadl代码描述（若aadl代码足以完成描述，禁止使用下文C代码格式），你可以将这些特殊条目通过调用C代码的格式实现，其格式为：
    ```
    Source_Language => (C); -- 源语言为，需在实现中连接函数
    Source_Name     => "function_name: xxxx"; -- 指定应连接的外部函数名
    Source_Text     => ("source_file: xxx.c");-- 指定该函数所在的源文件
    ```
- 以下是*原始需求*的内容：{req_slice}
</original_requirement>

<analysis_of_requirement>
# 需求的分析遵循以下规则：
- 按照字典列表逐条给出了对需求的细致分析
- 这些信息不一定完全正确，只作为你的生成参考，若出现冲突以*原始需求*为标准
- 以下是对需求分析：{analysis_data}
</analysis_of_requirement>

<summary_of_total_project>
# 项目组件架构的使用遵循以下规则：
- 该部分提供了项目整体使用的组件的概述以及层级/调用关系
- 该关系帮助你理解组件的包含/调用/连接关系
- 以下是对于项目整体的架构设计
{summary}
</summary_of_total_project>
{rag_prompt}

<output>
# 你将按照一下规则输出aadl代码：
- 务必确保语法的正确性，这是你的第一前提
- 严格遵循标准 AADL 语法（AS5506C）
- 输出内容使用以下结构标记进行包裹（注意有两组尖括号）：
   <<BEGIN_AADL>>：代码开始
   <<END_AADL>>：代码结束
- 禁止使用未声明过的属性/组件；允许调用Base_types等官方默认package，但必须通过with引用，且需要放置在public后
- 有些组件虽然并不在当前模块中声明，但由于其被引用，你需要进行简单声明，以确保代码直接通过OSATE语法检验
- 禁止创建多余的组件代码；不要输出多余解释，禁止在开始处标注`aadlversion 2;`，仅输出符合要求的代码，且必须使用英文
- 使用
```
package model
public
with xxx;
...
end model;
```
格式包裹整个package，其中model应当替换为本模块名称，xxx是可能引用的官方package
</output>

<syntax_helper>
1. 禁止使用未声明过的属性/组件；允许调用Base_types等官方默认package，但需要通过with引用
2. 禁止在开始处标注`aadlversion 2;`，仅输出符合要求的代码，且必须使用英文
3. 子程序调用的格式如下，且只能定义在组件的实现（implementation）中，注意分号的使用
```
        calls function_name:{{
             calc_sequence: subprogram Altitude_Calc_Program.Impl;}};
```
其中function_name和calc_sequence都是可替换的名称
4. 对于一些属性的赋值，若该属性值并非官方定义，使用
```
Communication_Properties::Protocol => "ISO11898_1";
```
形式描述，而不是其他
5. 所有的实现implementation，必须要在其前进行声明而不是直接使用
6. 有些组件虽然并不在当前模块中声明，但由于其被引用，你需要进行简单声明（例如a将b作为子组件，b应当被声明/实现），以确保代码直接通过OSATE语法检验
7. 组件声明中的features和组件实现（implementation）中的properties不能混用，有些属性或连接需要在这两种结构内各自使用
8. 空间大小使用Mbytes，KBytes，Bytes，bits等写法，Compute_Execution_Time的值是时间区间
9. 确保你使用的属性名称在aadl中真实存在
10. Dispatch_Protocol => Periodic;只能放在thread实现中
11. 若引用到了一个组件的implementation，则必须要有该组件的implementation而不是只有type
12. 若没有fearture/propreties，则不写这些关键词
13. subcomponents只能在implementation中定义
```
</syntax_helper>
"""
        code = self.llm.complete(prompt)
        try:
            splited_code = str(code).split("<<BEGIN_AADL>>")[1].split("<<END_AADL>>")[0]
            with open(f"./{base_path}/step3-build_aadl_code/{name_content}.aadl", "w", encoding="utf-8") as f:
                f.write(splited_code)
        except Exception as e:
            print("创建代码环节：在LLM输出切分时发生错误")
            print(e)
            with open(f"./{base_path}/step3-build_aadl_code/{name_content}_error.txt", "w", encoding="utf-8") as f:
                f.write(str(code))
        return code
    
    def single_check(self, base_path: str, req_slice: str) -> str:
        name_match = re.search(r"模块名称：\s*([\w_]+)", req_slice)
        name_content = name_match.group(1)
        # with open(f"./{base_path}/step3-build_aadl_code/{name_content}.aadl", "r", encoding="utf-8") as f:
            # original_code = f.read()
        result_file = f"./{base_path}/step3-build_aadl_code/{name_content}_result.txt"
        t0 = time.time()
        cmd = [
            self.inspector,
            "-a", f"./{base_path}/step3-build_aadl_code/{name_content}.aadl",
            "--plugin", "Static.parse",
            "--result", result_file,
            "--show", "false"
        ]
        print(cmd)
        cmd_result = subprocess.run(cmd, capture_output=True, text=True, shell=True, check=True)
        if os.path.exists(result_file) and os.path.getmtime(result_file) >= t0:
            print(f"检查完成，结果文件：./{base_path}/step3-build_aadl_code/{name_content}_result.txt")
            flag = True
        else:
            print("语法检查失败，未生成结果文件。")

    def check_aadl(self, base_path: str, req_slice: str):
        name_match = re.search(r"模块名称：\s*([\w_]+)", req_slice)
        name_content = name_match.group(1)
        with open(f"./{base_path}/step3-build_aadl_code/{name_content}.aadl", "r", encoding="utf-8") as f:
            original_code = f.read()
            new_code = original_code
        result_file = f"./{base_path}/step4-fix_aadl/{name_content}_result.txt"
        for attempt in range(self.max_attempt):
            with open(f"./{base_path}/step4-fix_aadl/{name_content}.aadl", "w", encoding="utf-8") as f:
                f.write(new_code)
            flag = False
            t0 = time.time()
            cmd = [
                self.inspector,
                "-a", f"./{base_path}/step4-fix_aadl/{name_content}.aadl",
                "--plugin", "Static.parse",
                "--result", result_file,
                # "--show", "false"
            ]
            print(cmd)
            cmd_result = subprocess.run(cmd, capture_output=True, text=True, shell=True, check=True)
            while not os.path.exists(result_file) or not os.path.getmtime(result_file) >= t0:
                print("等待检查文件生成")
                time.sleep(5)
            if os.path.exists(result_file) and os.path.getmtime(result_file) >= t0:
                print(f"检查完成，结果文件：./{base_path}/step4-fix_aadl/{name_content}.txt")
                flag = True
            else:
                print("语法检查失败，未生成结果文件。")
            """
            本地检查给出结果，三种处理模式
            """
            if not flag:
                # unparsed情况
                if attempt == 2:
                    break
                new_code = self.unparsed_code(base_path, req_slice)
            else:
                with open(result_file, "r") as f:
                    content = f.read()
                info = extract_error_between_markers(content)
                if "Model parsed successfully" in info:
                    # success情况
                    print("完成检查，代码位于./{base_path}/step4-fix_aadl/{requirement_name}.aadl")
                    with open(f"./{base_path}/log.txt", "a", encoding="utf-8") as log_file:
                        log_file.write(f"{name_content}模块: successfully get answer by {attempt+1} attempts\n\n")
                        return
                else:
                    if attempt == 2:
                        break
                    error_pattern = r'lappend lines\(error\) (\d+)\s+TextEditor::fastaddText \$sbpText "(.*?)"\s*?"'
                    matches = re.findall(error_pattern, info, re.DOTALL)
                    error_messages = []
                    for match in matches:
                        lines_error = match[0]
                        fastadd_text = match[1]
                        single_error_info = f"error in line {lines_error}, the reason is {fastadd_text}"
                        error_messages.append(single_error_info)
                    full_error_message = "\n".join(error_messages)
                    print(full_error_message)
                    new_code = self.error_code(base_path, req_slice, full_error_message)
        with open(f"./{base_path}/log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"{name_content}模块: failed to get answer by 10 attempts\n\n")
    
    def unparsed_code(self, base_path: str, req_slice: str) -> str:
        """
        with open(f"./{base_path}/step0-requirement/{requirement_name}.txt", "r", encoding="utf-8")as f:
            requirement_content = f.read()
        """
        name_match = re.search(r"模块名称：\s*([\w_]+)", req_slice)
        name_content = name_match.group(1)
        # requirement_content = f"<编号>: <{requirement_dict.get('需求ID')}>; <描述>: <{requirement_dict.get('需求描述')}>"
        with open(f"./{base_path}/step4-fix_aadl/{name_content}.aadl", "r", encoding="utf-8") as f:
            code_ori = f.read()
        unparsed_prompt = f"""
<background>
你是一名熟悉 AADL（Architecture Analysis and Design Language）语言标准的系统建模专家。现在你将获得一份需求文件，以及一份待修改的aadl文件。你的目的是：改正其中的错误，使得其能够被静态分析：
可能出现的错误包括：
（1）符号缺失：例如call name:{{...}}调用子程序后忘记在大括号后加分号
（2）错误的属性放置：例如modes不能放在implementation中，而只能放在声明里
（3）不存在的包引用
（4）其他语法错误，需要你进行鉴别
</background>

<input>
你将得到以下输入：
【原始需求】：{req_slice}
【待修正代码】：{code_ori}
</input>

<output>
你将按照以下要求输出结果：
- 首要目标是：确保输出符合AS5506C标准
- 次要目标是：确保符合需求需求语义，尽量少的改动代码
- 输出格式为：仅生成aadl代码，禁止输出任何其他的内容与分析
- 输出代码使用以下结构标记进行包裹（注意有两组尖括号）：
   <<BEGIN_AADL>>：表示代码开始
   <<END_AADL>>：表示代码结束
</output>

<syntax_helper>
1. 禁止使用未声明过的属性/组件；允许调用Base_types等官方默认package，但需要通过with引用
2. 禁止在开始处标注`aadlversion 2;`，仅输出符合要求的代码，且必须使用英文；禁止在代码前后通过```aadl进标记
3. 子程序调用的格式如下，且只能定义在组件的实现（implementation）中，且必须要放在该组件内部的首位，注意分号的使用
```
        calls function_name:{{
             calc_sequence: subprogram Altitude_Calc_Program.Impl;}};
```
其中function_name和calc_sequence都是可替换的名称
4. 对于一些属性的赋值，若该属性值并非官方定义，使用
```
Communication_Properties::Protocol => "ISO11898_1";
```
形式描述，而不是其他
5. 所有的实现implementation，必须要在其前进行声明而不是直接使用
6. 有些组件虽然并不在当前模块中声明，但由于其被引用，你需要进行简单声明，以确保代码直接通过OSATE语法检验
7. 组件声明中的features和组件实现（implementation）中的properties不能混用，有些属性或连接需要在这两种结构内使用
8. 若引用到了一个组件的implementation，则必须要有该组件的implementation而不是只有type
9. 若没有fearture/propreties，则不写这些关键词
10. subcomponents只能在implementation中定义
```
</syntax_helper>
"""
        response = self.llm.complete(unparsed_prompt)
        try:
            splited_response = str(response).split("<<BEGIN_AADL>>")[1].split("<<END_AADL>>")[0]
            """
            with open(f"./step4-fix_aadl/{requirement_name}.aadl", "r", encoding="utf-8") as f:
                f.write(splited_response)
            """
            return splited_response
        except Exception as e:
            print("重构unparse代码部分：LLM相应切割发生错误")
            print(e)

    def error_code(self, base_path: str, req_slice: str, error_info: str) -> str:
        """
        with open(f"./{base_path}/step0-requirement/{requirement_name}.txt", "r", encoding="utf-8")as f:
            requirement_content = f.read()
        """
        name_match = re.search(r"模块名称：\s*([\w_]+)", req_slice)
        name_content = name_match.group(1)
        # requirement_content = f"<编号>: <{requirement_dict.get('需求ID')}>; <描述>: <{requirement_dict.get('需求描述')}>"
        with open(f"./{base_path}/step4-fix_aadl/{name_content}.aadl", "r", encoding="utf-8") as f:
            code_ori = f.read()
        error_prompt = f"""
<background>
你是一名熟悉 AADL（Architecture Analysis and Design Language）语言标准的系统建模专家。现在你将获得一份需求文件，一个使用tk库编写的校验程序给出的语法错误信息，以及一份待修改的aadl文件。
</background>

<adaptive_thinking_framework>
你将按照以下步骤思考：
- 首先，依据错误报告确定错误信息与位置
- 其次，分析错误信息是否不够精确，依据错误位置的上下文进一步思考
- 最终，进行代码修改
</adaptive_thinking_framework>

<input>
你将得到以下信息：
【错误信息】：{error_info}
【原始需求】：{req_slice}
【待修正代码】：{code_ori}
</input>

<output>
按照如下规则输出修正后的代码：
- 首要目标是: 确保输出符合AS5506C标准，能够被读取
- 次要目标是: 确保符合需求需求语义，尽量少的改动代码
- 输出格式为：仅生成aadl代码，禁止输出任何其他的内容与分析
- 输出代码使用以下结构标记进行包裹（注意有两组尖括号）：
   <<BEGIN_AADL>>：表示代码开始
   <<END_AADL>>：表示代码结束
</output>

<syntax_helper>
1. 禁止使用未声明过的属性/组件；允许调用Base_types等官方默认package，但需要通过with引用，且必须放在public之后
2. 禁止在开始处标注`aadlversion 2;`，仅输出符合要求的代码，且必须使用英文；禁止在代码前后通过```aadl进标记
3. 子程序调用的格式如下，只能定义在组件的实现（implementation）中，且必须在组件实现部分率先定义。注意分号的使用
```
    calls function_name:{{
         calc_sequence: subprogram Altitude_Calc_Program.Impl;}};
```
其中function_name和calc_sequence都是可替换的名称
4. 对于一些属性的赋值，若该属性值并非官方定义，使用类似“包名::属性名 => "属性值"”的格式，例如下文的通信协议
```
Communication_Properties::Protocol => "name";
```
形式描述，而不是其他（其中name替换为你的属性值）
5. 所有的实现implementation，必须要在其前进行声明而不是直接使用
6. 有些组件虽然并不在当前模块中声明，但由于其被引用，你需要进行简单声明，以确保代码直接通过OSATE语法检验
7. 组件声明中的features和组件实现（implementation）中的properties不能混用，有些属性或连接需要在这两种结构内使用
8. *功能说明*和*具体行为*描述代码的功能用途，你需要通过aadl代码实现
仅当*功能说明*和*具体行为*某些条目无法用aadl代码描述（若aadl代码足以完成描述，禁止使用下文C代码格式），你可以将这些特殊条目通过调用C代码的格式实现，其格式为：
    ```
    Source_Language => (C); -- 源语言为，需在实现中连接函数
    Source_Name     => "function_name: xxxx"; -- 指定应连接的外部函数名
    Source_Text     => ("source_file: xxx.c");-- 指定该函数所在的源文件
    ```
9. 空间大小使用Mbytes，KBytes，Bytes，bits等写法，Compute_Execution_Time的值是时间区间
10. 确保你使用的属性名称在aadl中真实存在
11. 若引用到了一个组件的implementation，则必须要有该组件的implementation而不是只有type
12. 若没有fearture/propreties，则不写这些关键词
13. subcomponents只能在implementation中定义
</syntax_helper>
""" 
        response = self.llm.complete(error_prompt)
        print(response)
        try:
            splited_response = str(response).split("<<BEGIN_AADL>>")[1].split("<<END_AADL>>")[0]
            return splited_response
        except Exception as e:
            print("重构error代码部分：LLM相应切割发生错误")
            print(e)


def get_csv(file: str) -> List:
    df = pd.read_excel(file)
    req_list = []
    for index, row in df.iterrows():
        info = {
            "需求ID": row["需求ID"],
            "需求描述": row["需求描述"],
            "类型": row["类型"],
            "所属章节": row["所属章节"]
        }
        req_list.append(info)
    return req_list


def get_json(file: str) -> List:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    req_list = []
    for index, sub_req in enumerate(data):
        info = {
            "需求ID": f"{sub_req.get('需求id')}-{index}",
            "需求描述": sub_req.get("需求"),
            "类型": sub_req.get("类型"),
            "所属章节": sub_req.get("所属章节")
        }
        req_list.append(info)
    return req_list


def get_txt(file: str) -> List:
    with open(file, "r", encoding="utf-8") as f:
        data = f.read()
    modules = []
    current_module = []
    recording = False

    for line in data.splitlines():
        if line.strip() == "$module start$":
            recording = True
            current_module = []
            # current_module = [line]
        elif line.strip() == "$module end$":
            # current_module.append(line)
            modules.append('\n'.join(current_module))
            recording = False
        elif recording:
            current_module.append(line)

    return modules

if __name__ == "__main__":
    """
    AADL-inspector无法读取中文路径
    """
    my_agent = aadlAgent()
    # testing_example_name = "LAMP"
    # testing_example_name = "end_to_end"
    # testing_example_name = "pacemaker"
    # testing_example_name = "redundancy"
    # testing_example_name = "regulator"
    # testing_example_name = "ROSACE"
    # testing_example_name = "time_trigger"
    # testing_example_name = "satellite"
    # testing_example_name = "task_calling"
    # testing_example_name = "satety"
    # testing_example_name = "canbus"
    testing_example_name = "radar"
    base_path = f"data/{testing_example_name}"

    req_list = get_txt(f"{base_path}/output.txt")
    
    for req in req_list:
        match = re.search(r"模块名称：\s*([\w_]+)", req)
        if match:
            module_name = match.group(1)
            print("模块名称为：", module_name)
        else:
            print("输出格式有误，未找到模块名称")
        print("*" * 50)
        my_agent.analyse_req_details(base_path, req)
        my_agent.analyse_rules(base_path, req)
        my_agent.get_rules_reference(base_path, req)
        my_agent.build_aadl(base_path, req)
        # my_agent.single_check(base_path, req)
        my_agent.check_aadl(base_path, req)