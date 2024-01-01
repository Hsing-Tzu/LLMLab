# LLM LAB
**Members**
@Hsing-Tzu @kennyliou @imjessica @mason45ok


**11/2突然想到**
> [name=劉彥谷]
如果python的輸出是使用者無法預測，例如爬蟲回覆的結果我們無法預測PTT的所有標題，那我們就在得到第一次API_fetch時，印出執行結果讓使用者回覆是或否，以判斷error = 1 or error = 0。

**11/2突然想到 Part 2**
> [name=劉彥谷]
我們程式的過程是不是就是一個fine-tune的過程，那我們是不是可以同時把json生出來

## 11/1｜First Discussion
### Topic 
解決自己在學習Python時，當我們詢問ChatGPT相關問題時，會得到不準確的答案，我們希望能使其答出準確的答案，因此，我們希望ChatGPT產出程式碼之後，可以先編譯過，確認過執行結果是正確的，再回傳程式碼。
### Function
* API fetch
* json處理
    * 尋找兩對三個「 ` 」所包含的程式碼，並且刪除不必要的符號
* 編譯
    * eval
* 檢查
    * eval 結果是否等於測資，是：回傳python、否：重複執行
### 程式執行流程
#### 這是一個蘇都扣
situation = 0;
```python=
while True:
    if situation == 1:
        #call API
    
    #製作一個test.py
    #執行test.py
    #接收回傳結果
```
#### 工具
```python=
# 設定要執行的Python腳本文件
command = "python try.txt"

# 使用subprocess執行命令
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# 等待命令完成
stdout, stderr = process.communicate()

# 檢查命令是否成功
if process.returncode == 0:
    print("命令成功執行，輸出為：")
    print(stdout)
else:
    print("命令執行失敗，錯誤訊息為：")
    print(stderr)

```
寫入GPT輸出至txt
```python=
user_result = '''
#python 
from bs4 import BeautifulSoup as bs
import requests

url="https://www.ptt.cc/bbs/Stock/index6651.html"
headers={
    "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
}

res = requests.get(url , headers=headers)
print(res.text)'''
path = 'python try'
f = open(path, 'w')
f.write(str(user_result))
f.close()
```
### 最終程式碼
https://github.com/Hsing-Tzu/LLMLab/blob/main/LLMLAB_1101.ipynb
```python
user_result = "Hello World"
user_question = "1Can you give me the python code which can print Hello World?"

# prompt = ""+ user_question + "" + "，且預期結果為" + user_result
```
```python
import requests
import subprocess

def CALL_API (user_question):
    
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    API_fetch = query({
        "inputs": user_question,
    })

    return API_fetch

API_fetch = CALL_API(user_question)
print(API_fetch)
```
```python
import re

def json_processed (API_fetch):
    text = API_fetch[0]['generated_text']
    code_blocks = re.findall(r'```\n(.*?)\n```', text, re.DOTALL)

    
    return block

json_processed = json_processed(API_fetch)
```
```python
situation = 1 #0成功、1失敗
error = 0
```
```python
while situation == 1:
    if error == 1:
        new_API_fetch = CALL_API(stderr)
        print(new_API_fetch)
        
    #製作一個test.txt
    path = 'test.txt'
    f = open(path, 'w')
    f.write(str(json_processed))
    f.close()
    
    #執行test.txt
    # 設定要執行的Python腳本文件
    command = "python test.txt"

    # 使用subprocess執行命令
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 等待命令完成
    stdout, stderr = process.communicate()
    stdout = stdout.strip()
    #print(stdout)

    
    # 檢查命令是否成功
    if process.returncode == 0 and stdout == user_result:
        print("命令成功執行，輸出為：")
        situation = 0
        print(stdout)
    else:
        print("命令執行失敗，錯誤訊息為：")
        situation = 1
        error = 1
        stderr = stderr + "This is not the result I want, the result I want is " + user_result + ". Can you generate a new code?"
        print(stderr)
```
## 11/6｜Pre-Discussion
https://colab.research.google.com/drive/1ZveGLd_uMhTei-EAtOuOpsnPVgh7Bu3k?usp=sharing
### Fine-tuning
#### Upload a training file
```python
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.File.create(
  file=open("mydata.jsonl", "rb"),
  purpose='fine-tune'
)
```
### Create a fine-tuned model
```python
openai.FineTuningJob.create(training_file="file-abc123", model="gpt-3.5-turbo")
```
### Fine-tuning GPT-2
#### Test in huggingface
```python
#gpt-2
import requests

API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": "Bearer {API_KEY}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

output = query({
	"inputs": "Can you generate a python code to make a bar chart based on matplotlib?",
})
```
#### Test in Colab
```python
pip install transformers
```
```python
pip install torch
```
```python
from google.colab import drive
drive.mount('/content/drive')
f = open('/content/drive/MyDrive/Colab Notebooks/test.txt','r')
a = f.read()
print(a)
f.close()
```
```python
import accelerate
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 加载预训练模型和分词器
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 准备数据集
train_path = '/content/drive/MyDrive/Colab Notebooks/train.txt'
test_path = '/content/drive/MyDrive/Colab Notebooks/test.txt'

train_dataset = TextDataset(
  tokenizer=tokenizer,
  file_path=train_path,
  block_size=128
)

test_dataset = TextDataset(
  tokenizer=tokenizer,
  file_path=test_path,
  block_size=128
)

data_collator = DataCollatorForLanguageModeling(
  tokenizer=tokenizer, mlm=False
)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=400,
    save_steps=800,
    warmup_steps=500,
    prediction_loss_only=True,
)

# 初始化训练器并开始微调
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

```
## 11/7｜Meeting with Dr.Tsai
### word2vec
https://www.tensorflow.org/text/tutorials/word2vec
https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/word2vec.ipynb

### BERT
https://mofanpy.com/tutorials/machine-learning/nlp/cbow

### 方向確認
應該先embedding 再來考慮fine-tuning

## 11/8｜Embedding
https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/videos/semantic_search.ipynb
### 將句子編碼成嵌入向量
:::spoiler 將句子編碼成嵌入向量

```
使用預訓練的語言模型==將句子編碼成嵌入向量==
```

這行代碼導入了PyTorch庫，這是一個在Python中實現深度學習的流行庫。

```python
import torch
```
從transformers庫中導入AutoTokenizer和AutoModel。這個庫提供了許多預訓練的模型，用於自然語言處理任務。
```python
from transformers import AutoTokenizer, AutoModel
```
定義了一個列表sentences，包含三個==要處理的英文句子==。
```python
sentences = [
    "I took my dog for a walk",
    "Today is going to rain",
    "I took my cat for a walk",
]
```
設置一個變量model_ckpt，它包含了==預訓練模型的名稱==，用於後續加載模型和分詞器。
```python
model_ckpt = "sentence-transformers/all-MiniLM-L6-v2"
```
使用model_ckpt中指定的模型名稱創建一個分詞器實例，這個分詞器將用於將句子轉換成==模型能理解的格式==。
```python
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
```
加載與分詞器相對應的語言模型，這個模型將用來==生成嵌入向量==。
```python
model = AutoModel.from_pretrained(model_ckpt)
```
調用分詞器將句子列表轉換成模型需要的輸入格式，並將它們==轉換成PyTorch張量==。這裡還設置了填充（以使所有句子長度一致）和截斷（以避免超長句子）。
```python
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
```
在不計算梯度的情況下執行模型（這在推論時節省計算資源），並將編碼後的輸入提供給模型，==得到模型的輸出==。
```python
with torch.no_grad():
    model_output = model(**encoded_input)
```
從模型輸出中取出最後一層隱藏狀態，這代表了==輸入句子中每個token的嵌入向量==。
```python
token_embeddings = model_output.last_hidden_state
```
打印出token嵌入的形狀，這會顯示出==張量的維度==，通常是（==句子數量, 每句話的token數量, 嵌入維度==）。
這告訴我們模型產生了多少個嵌入向量，以及每個向量的大小。
```python
print(f"Token embeddings shape: {token_embeddings.size()}")
```
Output: ```Token embeddings shape: torch.Size([3, 9, 384])```

:::
### 產生句子級的嵌入並標準化

:::spoiler 產生句子級的嵌入並標準化
```
定義mean_pooling：處理由transformer模型（如BERT或相似模型）生成的token嵌入，並且產生句子級的嵌入。

接著，它標準化這些嵌入並打印出嵌入的形狀。
```

這行代碼==導入PyTorch深度學習庫中的functional模組==，通常簡稱為F。這個模組包含了一組函數，這些函數可以在==不需要定義完整神經網絡層的情況下，直接對數據進行操作==。
```python
import torch.nn.functional as F
```
定義一個名為mean_pooling的函數
```python
# 它接受模型輸出和注意力遮罩（attention mask）作為參數。
def mean_pooling(model_output, attention_mask):
    
    #從模型輸出中提取最後一層隱藏狀態，這包含了句子中每個token的嵌入表示。
    token_embeddings = model_output.last_hidden_state
    
    #這行代碼將注意力遮罩展開到與token嵌入相同的尺寸。unsqueeze(-1)在最後一個維度上增加一個軸，使得遮罩可以通過expand方法延展到與嵌入相同的形狀。
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    
    # 將擴展後的遮罩應用於token嵌入，然後對每個句子進行求和，得到加權的token嵌入總和。這個總和通過每個句子的非零token數量進行歸一化，使用torch.clamp來避免除以零的情況，最小值設定為1e-9。
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

```
這行代碼調用mean_pooling函數，==將模型輸出和編碼輸入中的注意力遮罩作為參數傳入==，得到==每個句子的嵌入==。
```python
sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
```
使用PyTorch的functional API中的normalize函數對句子嵌入進行==L2歸一化==，這有助於在後續的任務中改善模型的性能和穩定性。
```python
# Normalize the embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
```
最後，這行代碼打印出==經過歸一化處理的句子嵌入向量的形狀==。這會顯示張量的維度，通常是（==句子數量, 嵌入向量維度==）。
這告訴我們經過處理後得到了==多少個句子嵌入、以及每個嵌入的大小==。
```python
print(f"Sentence embeddings shape: {sentence_embeddings.size()}")
```
Output: ```Sentence embeddings shape: torch.Size([3, 384])```
:::

### 計算一組句子嵌入之間的餘弦相似度
:::spoiler 計算一組句子嵌入之間的餘弦相似度
這行代碼導入了==NumPy==庫，這是Python中用於進行科學計算的基礎庫，提供了大量的數學函數和多維陣列操作。
```python
import numpy as np
```
從sklearn.metrics.pairwise模組中導入cosine_similarity函數。sklearn（Scikit-learn）是一個提供許多常見==機器學習算法的Python庫==，cosine_similarity用於計算向量之間的==餘弦相似度==。
```python
from sklearn.metrics.pairwise import cosine_similarity
```
這行代碼將PyTorch張量中的==句子嵌入轉換成NumPy陣列==。detach()方法將嵌入從當前計算圖中分離出來，這樣可以防止在轉換成NumPy陣列時發生梯度信息的錯誤傳遞。
```python
sentence_embeddings = sentence_embeddings.detach().numpy()
```
初始化一個二維陣列scores，用零填充，其形狀由句子嵌入的數量決定。
這個陣列將用來==儲存所有句子嵌入之間的餘弦相似度分數==。
```python
scores = np.zeros((sentence_embeddings.shape[0], sentence_embeddings.shape[0]))
```
開始一個循環，==對每個句子嵌入進行迭代==。
```python!
for idx in range(sentence_embeddings.shape[0]):
    
    # 對於每個句子嵌入，計算它與所有其他句子嵌入的餘弦相似度。
    # cosine_similarity函數接受兩個參數：一個是單個句子嵌入（需要包裝在列表中，因為cosine_similarity期望二維陣列），另一個是整個句子嵌入陣列。
    # 函數返回一個一維陣列，包含了當前句子嵌入與陣列中每個句子嵌入的相似度分數，這些分數被賦值給scores陣列中對應的行。
    
    scores[idx, :] = cosine_similarity([sentence_embeddings[idx]], sentence_embeddings)[0]
```
:::
### 獲取文本的嵌入向量
::: spoiler 獲取文本的嵌入向量
```
這段程式碼使用了datasets庫來加載和預處理SQuAD（Stanford Question Answering Dataset）數據集，並且定義了一個函數來獲取文本的嵌入向量。
```
這行代碼導入了datasets庫的load_dataset函數，這個函數用於加載和處理公開可用的數據集。
```python
from datasets import load_dataset
```
加載SQuAD數據集的驗證集部分，使用種子42進行隨機打亂，然後選擇前100個樣本。shuffle是為了確保選擇的是隨機的樣本，而select則是從打亂後的數據集中選擇一個子集。
```python
squad = load_dataset("squad", split="validation").shuffle(seed=42).select(range(100))
```
定義了一個函數get_embeddings，它接受一個文本列表作為參數。
```python
def get_embeddings(text_list):
    
    #使用先前提到的分詞器將文本列表轉換為模型需要的格式，進行填充和截斷以保持一致長度，並將轉換後的輸入轉換為PyTorch張量。
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    
    #這行代碼將分詞器輸出的字典進行了解包，它實際上沒有改變任何內容，可能是為了清晰表示輸入的結構或是修復某些環境下的相容性問題。
    encoded_input = {k: v for k, v in encoded_input.items()}
    
    #在無需計算梯度的上下文中執行模型，這是推論階段常見的做法，可以節省記憶體和計算時間。
    with torch.no_grad():
        model_output = model(**encoded_input)
        
    #返回使用mean_pooling函數處理的模型輸出，該函數產生句子級的嵌入。
    return mean_pooling(model_output, encoded_input["attention_mask"])

```
使用map函數遍歷SQuAD數據集的每一個樣本，對於每個樣本的context字段，使用get_embeddings函數計算嵌入向量。計算後，將嵌入從PyTorch的CUDA張量轉換到CPU張量，再將其轉為NumPy陣列，並將結果存儲在新的字段embeddings中。這個過程將為SQuAD數據集的每個樣本添加對應的嵌入向量。

```python
squad_with_embeddings = squad.map(
    lambda x: {"embeddings": get_embeddings(x["context"]).cpu().numpy()[0]}
)
```
:::
### Last
:::spoiler Last
```
這段程式碼對包含嵌入向量的SQuAD數據集建立了一個FAISS索引，以便快速進行高效的相似性搜索，然後對一個特定的問題進行編碼並檢索最相關的數據集條目。這段程式碼對包含嵌入向量的SQuAD數據集建立了一個FAISS索引，以便快速進行高效的相似性搜索，然後對一個特定的問題進行編碼並檢索最相關的數據集條目。
```
:::

### w3school python 爬蟲
暫時只爬出其中一頁(https://www.w3schools.com/python/python_getstarted.asp)

```python
from bs4 import BeautifulSoup 
import requests
import re
import pandas as pd

url="https://www.w3schools.com/python/python_getstarted.asp"
headers={
    "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
}

res = requests.get(url, headers=headers)
soup = BeautifulSoup(res.text, "html.parser")
sample_t=[]
# 抓取內容
sample = soup.find_all("div", class_="w3-example")
container_div = soup.find("div", class_="w3-col l10 m12")
if container_div:
    p_elements = container_div.find_all(['p', 'li'])
    p_elements_t = "\n".join([content.get_text() for content in p_elements])
for div in sample:
    text = div.text
    index = text.find(">>>")
    if index != -1:
        text = text[index + 3:]

    # 使用正则表达式去除换行和空白
    text = re.sub(r'\s+', ' ', text)
    sample_t.append(text)
print(p_elements_t)
print("-"*50)
sample_t
```

### embedding code
[colab URL](https://colab.research.google.com/drive/1tKJ6cyL9y6Y8Fbtatcp1PhTcA8RDPdYO?usp=sharing)*使用師大帳號、建立副本*

[Three little pigs](https://github.com/gbishop/samisays/blob/master/samisays/three%20little%20pigs.txt)
[Harry Potter](https://github.com/bobdeng/owlreader/blob/master/ERead/assets/books/Harry%20Potter%20and%20The%20Half-Blood%20Prince.txt)

### embedding adjust
在 Word2Vec 模型中，各個可調整變量的範圍和功能如下：

- `vector_size`：==詞向量的維度==，可以是任何正整數。
    - 增加維度可以提高模型的複雜度和容量，但也可能導致過擬合，特別是在詞彙量不大的數據集上。通常範圍在 50 到 300 之間。

- `window`：==上下文窗口大小==，也是正整數。窗口大小決定了在考慮目標詞的上下文時要考慮的周圍詞語的範圍。
    - 窗口大小較小可能會讓模型學習到更多關於詞語特定用途的信息，而較大的窗口則有助於學習詞語的廣泛用途。一般來說，窗口大小設置在 5 到 20 之間。

- `min_count`：==詞頻下限==，決定了詞必須出現的最小次數才能被考慮進詞彙表。這是一個非負整數。
    - 將此值設置得太低可能會導致許多罕見詞污染詞向量空間，設置得太高可能會忽略有用的信息。通常設置為 1 到 100。

- `workers`：==進行訓練的線程數量==，這個數字應該和你的機器的處理器核心數相符。這是一個正整數，如果你的機器有多個核心，增加這個值可以加快訓練速度。
    - 然而，這並不會影響訓練後模型的性能或質量。

這些變量可以根據具體的數據集和應用場景進行調整，以獲得最佳的模型表現。例如，較小的數據集可能需要較小的 `vector_size` 和較高的 `min_count`，而較大的數據集則可能需要較大的 `vector_size` 和較小的 `min_count`。調整這些參數時，可能需要通過交叉驗證等方法來找到最優的參數組合。

## 11/13｜Pre-Discussion
https://hackmd.io/@meebox/SJ7Um4Whs

### Selenium Web Crawler for embedding data
:::spoiler Code
https://github.com/knyliu/LATIA112-1/blob/main/LATIA_HW2
:::
### Scrapy for embedding data
:::spoiler Code
https://github.com/knyliu/LATIA112-1/blob/main/LATIA_HW2

Create a project in terminal:
```
scrapy startproject w3schools_scrapy
```
Start a project in terminal:
```
scrapy crawl w3schools
```
"w3schools" is depended upon for the name in the Python code.
:::
---
> [name=劉彥谷]
> @mason45ok 寫的一頁爬蟲，不但只能爬一頁、單一頁又爬不完整。

> [name=mason45ok]
> 我很抱歉😢

:::warning
To Be Discussed:
1. 有些資料（頁面）在爬蟲時會被跳過
    例如：python try...except
2. 有些頁面底下還有更多頁面
    例如：Python String。
3. 有些頁面的內容會重複存到csv file中
4. 後面的頁數即便XPath是正確的（在網頁中存在此Xpath），卻也無法爬蟲。迴圈有寫檢測機制，會報錯：Element not found for Xpath。
    不確定是不是被擋？但又感覺不是。
:::
### BERT
https://mofanpy.com/tutorials/machine-learning/nlp/bert
> [name=劉彥谷]
> We can do it after Mason finishes the web crawler.

## 11/15｜BERT Model Trainning
要從頭開始訓練一個空的BERT模型，您需要遵循以下步驟：

1. **數據準備**：BERT的訓練需要大量的文本數據。這些數據通常是未標記的，來自各種來源，例如書籍、網站和新聞文章。您需要收集並預處理這些數據，包括分詞（Tokenization）、去除特殊字符、統一大小寫等。

2. **選擇預訓練任務**：BERT通常通過兩種預訓練任務來訓練：掩碼語言模型（Masked Language Model, MLM）和下一句預測（Next Sentence Prediction, NSP）。在MLM中，隨機地從輸入句子中遮蔽一些單詞，然後讓模型預測這些單詞。在NSP中，模型需要預測兩個句子是否連續。

3. **建立模型架構**：BERT是一個基於Transformer的模型，具有多層自注意力（Self-Attention）層。您需要根據需要選擇合適的層數、隱藏單元數、頭的數目等參數。

4. **設置超參數**：您需要選擇適合您數據和硬件的學習率、批大小（Batch Size）、訓練周期（Epochs）等。

5. **訓練模型**：使用您的數據和設置，開始訓練過程。這通常需要使用GPU或TPU等高性能計算資源。

6. **評估和調整**：在訓練過程中和之後，評估模型性能，並根據需要進行調整。這可能包括調整超參數、添加更多數據或更改預處理步驟。

7. **微調**：一旦模型在預訓練任務上表現良好，您可以通過在特定任務（如情感分析、問答等）上的進一步訓練來微調它。

請注意，訓練BERT是一個資源密集和時間密集的過程，需要大量的計算資源和時間。如果您沒有足夠的資源，可以考慮使用現成的預訓練模型進行微調，而不是從頭開始訓練。

:::spoiler BERT Code
```python
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=512):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                batch_encoding = tokenizer(line, add_special_tokens=True, truncation=True, max_length=self.block_size, return_token_type_ids=False)
                self.examples.append(batch_encoding.input_ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])

# 初始化tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # 如果您的数据是中文

# 加载数据集
file_path = 'harry_potter_1.txt'  # 您的数据文件路径
dataset = TextDataset(tokenizer, file_path)

# 创建数据collator，用于动态padding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# 配置模型
config = BertConfig.from_pretrained("bert-base-uncased")  # 根据您的需求选择适当的预训练模型
model = BertForMaskedLM(config)

# 训练参数设置
training_args = TrainingArguments(
    output_dir="./bert_output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# 开始训练
trainer.train()

```
:::

## 11/20｜Pre-Discussion


|   |11/22| 11/29 | 12/6 |
| -------- | -------- | -------- |-------- |
| 主要進度     | Get the Model| Test the model     | PPT slides     |
| Note |      ||Group Name|
| Daisy     |Study how to test the model|      | Text     |
| Kenny     |Clean the python data 11/27 Morning|      | Text     |
| Mason     |Study code for trainning model and train the model for python |      | Text     |
| Jessica     |Study code for trainning model and train the model for python |      | Text     |

**11/22**
* Online Disscussion
    * 環境set up 
    * Check the final code for trainning model
    * code for trainning the model
    * train harry potter
* Home
    * clean the python data *1 (Mon. Morning 11/27)
    * Study code for trainning model and train the model for python *2
        * Kenny CANNOT train the model because of his terrible MacBook Pro.
    * Study how to test the model *1
## 11/22｜環境建置
[tensorflow參考網址](https://discuss.tensorflow.org/t/tensorflow-not-detecting-gpu/16295/6)

**大家都至少想一個Group Name**

|  | Group Name 1 | Group Name 2 |
| -------- | -------- | -------- |
| Daisy     |   MDKJ (Mega Dialog Knowledge Jockey)   |   |
| Kenny     |      |      |
| Mason     |      |      |
| Jessica   |CodePioneers(CP)    |      |
## 11/27｜Pre-Disscussion
[Bert-問答](https://medium.com/analytics-vidhya/question-answering-system-with-bert-ebe1130f8def)  
[Bert-介紹](https://github.com/IKMLab/course_material/blob/master/bert-huggingface.ipynb)

### Problems we need to solve
* How do we get the model?(Or do we really need the model?)
    * word2Vec - 從空的Model開始train
        * We don't have that much data
    * BERT - 透過已有的model繼續下去train
        * But how about the code?
    * fine-tunning - 微調
        * Not we want
* 專題方向
    * 原本想解決：提問後會發散（且可以根據情境回答）
    * 現在的問題：可能沒辦法得到正確回答（不能生成code）
        * 那我們需要調整專題方向嗎？
        * 即便我們用BERT成功把model生出來，我們可能也得不到正確回覆
            * 也不一定可以透過情境問答
        * How about 用 GPT-2加上python data去微調？
            * 我們的創新？或許是我們可以自我檢測回覆是否是用戶所希望，直到相同，才回傳程式碼給用戶

## 11/29｜BERT Code and PPT Discussion
19:00 - 20:00 Serch and study the code

21:00 - XX:00 Discuss the PPT and Group name

> [name=Daisy]
train from scratch BERT

[BERT_from_scratch](https://github.com/antonio-f/BERT_from_scratch/blob/main/BERT_from_scratch.ipynb)

[Finetune a Pre-trained Model](https://huggingface.co/docs/transformers/training)

:::spoiler finetuning with asking question
```python=
def get_answer_using_bert(question, reference_text):
  
  bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

  bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

  input_ids = bert_tokenizer.encode(question, reference_text)
  input_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)

  sep_location = input_ids.index(bert_tokenizer.sep_token_id)
  first_seg_len, second_seg_len = sep_location + 1, len(input_ids) - (sep_location + 1)
  seg_embedding = [0] * first_seg_len + [1] * second_seg_len

  model_scores = bert_model(torch.tensor([input_ids]), 
  token_type_ids=torch.tensor([seg_embedding]))
  ans_start_loc, ans_end_loc = torch.argmax(model_scores[0]), torch.argmax(model_scores[1])
  result = ' '.join(input_tokens[ans_start_loc:ans_end_loc + 1])

  result = result.replace('#', '')
  return result

reference_text = 'Mukesh Dhirubhai Ambani was born on 19 April 1957 in the British Crown colony of Aden (present-day Yemen) to Dhirubhai Ambani and Kokilaben Ambani. He has a younger brother Anil Ambani and two sisters, Nina Bhadrashyam Kothari and Dipti Dattaraj Salgaonkar. Ambani lived only briefly in Yemen, because his father decided to move back to India in 1958 to start a trading business that focused on spices and textiles. The latter was originally named Vimal but later changed to Only Vimal His family lived in a modest two-bedroom apartment in Bhuleshwar, Mumbai until the 1970s. The family financial status slightly improved when they moved to India but Ambani still lived in a communal society, used public transportation, and never received an allowance. Dhirubhai later purchased a 14-floor apartment block called Sea Wind in Colaba, where, until recently, Ambani and his brother lived with their families on different floors.'
question = 'What is the name of mukesh ambani brother?'

get_answer_using_bert(question, reference_text)
```
:::

### 使上述的 PyTorch 程式碼使用 GPU 進行計算
要使上述的 PyTorch 程式碼使用 GPU 進行計算，您需要確保幾件事情：

1. **確認 CUDA 可用**: 首先，您需要檢查 CUDA 是否在您的系統上可用。可以使用 PyTorch 的 `torch.cuda.is_available()` 函數來進行這一檢查。

2. **指定計算裝置**: 如果 CUDA 可用，您需要指定一個裝置（device），將其設為 `"cuda"`。否則，您可以回退到使用 CPU。

3. **將模型和數據移至 GPU**: 您需要將您的模型和所有的輸入數據（如 `input_ids`, `attention_mask`, `labels`）移到 GPU 上。

以下是對您的程式碼進行這些修改的範例：

```python
import torch
from torch import nn
from tqdm import tqdm

# 檢查 CUDA 是否可用並設置裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型初始化和優化器設置
model = ...  # 替換為您的模型
optim = ...  # 替換為您的優化器
model.to(device)  # 將模型移至設定的裝置

epochs = 2

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()

        # pull all tensor batches required for training
        # 確保所有輸入數據也移至 GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # process
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        # extract loss
        loss = outputs.loss

        # calculate loss for every parameter that needs grad update
        loss.backward()

        # update parameters
        optim.step()

        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
```

請確保在使用 CUDA 之前，您的系統已經安裝了適當的 NVIDIA 驅動程式和 CUDA 工具包。此外，您的 PyTorch 版本也需要支持 CUDA。
![image](https://hackmd.io/_uploads/B1qd0jESa.png)

---
**什麼是PyTorch/tensor flow**
> 一個別人寫好的東西可以把Python寫給CPU算的程式改成用GPU計算


**CUDA?**
> GPU最小的單位叫做CUDA，是一個硬體單位，類似核心的概念。

**所以總而言之要把計算的方式改為GPU，該如和設定？**
>GPU要在以下的網址中才可以成功運作 > https://developer.nvidia.com/cuda-gpus
>
> 要確定CUDA版本、Pytorch的版本一致。
> 
> CUDNN軟體確定你是什麼單位（研究單位）
>
>下載CUDA，CUDNN解壓縮，裡面的東西拖進CUDA對應資料夾 => 環境設定完成


---

|  | To Do |  |
| -------- | -------- | -------- |
| Daisy     |      |      |
| Kenny     |      |      |
| Mason     |   Teach Us All About GPU Computing    |      |
| Jessica   |      |      |

## PPT
* 隊伍介紹
    * 一個人想一個自我介紹
        * 扮演什麼角色
    * 團隊名稱
        * 為什麼叫這個團名
        * 團隊理念、團隊介紹
* 創作理念
    * 為什麼要做這個專題？
        * 想解決什麼？
        * 目標用途
        * 目標族群
* 成果說明
    * 自動測試(擷取程式碼去問問題)
    * 模型
* BERT vs Fine-tunning
* 學習、心路歷程
    1. word2Vec
    2. BERT from Scratch
    3. Fine-tuning a Pre-trained Model
* 程式說明
    * 自動測試(擷取程式碼去問問題)
    * 模型
        * word2Vec
        * BERT from Scratch
        * Pre-train
        * Fine-tuning
* 其他補充
    * 未來會做到什麼(GPT-2)

## 12/13｜First Discussion

:::spoiler BERT from scratch程式碼
```python=
#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.36.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mlm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=model_args.trust_remote_code)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=[text_column_name],
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset line_by_line",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=[text_column_name],
                )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on every text in dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with training_args.main_process_first(desc="grouping texts together"):
            if not data_args.streaming:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
            else:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)

    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
```
:::

! pip install transformers==4.36.0.dev0

ERROR: Ignored the following yanked versions: 4.14.0, 4.25.0
ERROR: Could not find a version that satisfies the requirement transformers==4.36.0.dev0 (from versions: 0.1, 2.0.0, 2.1.0, 2.1.1, 2.2.0, 2.2.1, 2.2.2, 2.3.0, 2.4.0, 2.4.1, 2.5.0, 2.5.1, 2.6.0, 2.7.0, 2.8.0, 2.9.0, 2.9.1, 2.10.0, 2.11.0, 3.0.0, 3.0.1, 3.0.2, 3.1.0, 3.2.0, 3.3.0, 3.3.1, 3.4.0, 3.5.0, 3.5.1, 4.0.0rc1, 4.0.0, 4.0.1, 4.1.0, 4.1.1, 4.2.0, 4.2.1, 4.2.2, 4.3.0rc1, 4.3.0, 4.3.1, 4.3.2, 4.3.3, 4.4.0, 4.4.1, 4.4.2, 4.5.0, 4.5.1, 4.6.0, 4.6.1, 4.7.0, 4.8.0, 4.8.1, 4.8.2, 4.9.0, 4.9.1, 4.9.2, 4.10.0, 4.10.1, 4.10.2, 4.10.3, 4.11.0, 4.11.1, 4.11.2, 4.11.3, 4.12.0, 4.12.1, 4.12.2, 4.12.3, 4.12.4, 4.12.5, 4.13.0, 4.14.1, 4.15.0, 4.16.0, 4.16.1, 4.16.2, 4.17.0, 4.18.0, 4.19.0, 4.19.1, 4.19.2, 4.19.3, 4.19.4, 4.20.0, 4.20.1, 4.21.0, 4.21.1, 4.21.2, 4.21.3, 4.22.0, 4.22.1, 4.22.2, 4.23.0, 4.23.1, 4.24.0, 4.25.1, 4.26.0, 4.26.1, 4.27.0, 4.27.1, 4.27.2, 4.27.3, 4.27.4, 4.28.0, 4.28.1, 4.29.0, 4.29.1, 4.29.2, 4.30.0, 4.30.1, 4.30.2, 4.31.0, 4.32.0, 4.32.1, 4.33.0, 4.33.1, 4.33.2, 4.33.3, 4.34.0, 4.34.1, 4.35.0, 4.35.1, 4.35.2, 4.36.0)
ERROR: No matching distribution found for transformers==4.36.0.dev0

---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
Untitled-1.ipynb Cell 9 line 2
      1 # Will error if the minimal version of Transformers is not installed. Remove at your own risks.
----> 2 check_min_version("4.36.0")
      4 require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
      6 logger = logging.getLogger(__name__)

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\transformers\utils\__init__.py:241, in check_min_version(min_version)
    239     error_message = f"This example requires a minimum version of {min_version},"
    240 error_message += f" but the version found is {__version__}.\n"
--> 241 raise ImportError(
    242     error_message
    243     + "Check out https://github.com/huggingface/transformers/tree/main/examples#important-note for the examples corresponding to other "
    244     "versions of HuggingFace Transformers."
    245 )

ImportError: This example requires a minimum version of 4.36.0, but the version found is 4.35.1.
Check out https://github.com/huggingface/transformers/tree/main/examples#important-note for the examples corresponding to other versions of HuggingFace Transformers.


how to draw bar chart in python

My grandmother is an illiterate farmer, and I hope to let her know that her vegetable sales are increasing every year. In 2019, she sold 100 kg, in 2020 she sold 120 kg, in 2021 she sold 140 kg, in 2022 she sold 160 kg, and in 2023 she sold 200 kg.

How to get the data from MySQL using Python?

## 12/26｜Sprint Week Day 1
### 我們還缺什麼？
* BERT from scratch and result
* Fine-tunning code and result
    * 把book記在腦子裡面
* RAG test with pdf file
    * 考試open book
* PPT

https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891

---

[BERT from Scratch Tutorial - 很清楚](https://coaxsoft.com/blog/building-bert-with-pytorch-from-scratch)
```
[CLS] My cat mice likes to sleep and does not like [MASK] mice jerry
[SEP]
[CLS] jerry is treated as my pet too [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
```
https://huggingface.co/daryl149/llama-2-13b-chat-hf

### Fine-Tuning Code
#### Code
:::spoiler Code
```python=
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# llama-2-13b-chat-hf

# 載入預訓練的模型和分詞器
model_name = "LLaMA-model-name"  # 替換為實際的 LLaMA 模型名稱
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 為微調準備你的數據集
# 這應該是包含你希望模型學習的文本的數據集
dataset = "path_to_your_dataset"  # 替換為你數據集的路徑

# 定義一個方法來預處理數據
def preprocess_data(data):
    # 在這裡實現預處理步驟，如分詞等
    pass

# 預處理你的數據集
processed_dataset = preprocess_data(dataset)

# 定義你的訓練參數
training_args = {
    "num_train_epochs": 3,   # 訓練週期數
    "per_device_train_batch_size": 8,  # 每個設備的訓練批次大小
    "save_steps": 10_000,    # 每10,000步保存一次檢查點
    "weight_decay": 0.01,    # 權重衰減的強度
    # 根據需要添加更多參數
}

# 初始化 Trainer
from transformers import Trainer, TrainingArguments

training_arguments = TrainingArguments(**training_args)
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=processed_dataset,
    # 根據需要添加更多參數
)

# 開始微調
trainer.train()

# 保存微調後的模型
model.save_pretrained("path_to_save_your_model")
```
:::

```
your_project_folder/
│
├── model/                 # 預訓練模型檔案
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── vocab.json
│   └── merges.txt
│
├── data/                  # 訓練數據集
│   └── your_dataset.txt
│
├── scripts/               # 微調腳本
│   └── fine_tune.py
│
├── requirements.txt       # 環境配置檔案
│
└── [其他輔助腳本]

```
#### 如果你的資料在本機端：
:::spoiler 如果你的資料在本機端
```python=
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 指定本地模型的路徑
local_model_path = "path_to_your_local_model"  # 替換為你本地模型的路徑

# 載入本地的模型和分詞器
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# 為微調準備你的數據集
# 這應該是包含你希望模型學習的文本的數據集
dataset = "path_to_your_dataset"  # 替換為你數據集的路徑

# 定義一個方法來預處理數據
def preprocess_data(data):
    # 在這裡實現預處理步驟，如分詞等
    pass

# 預處理你的數據集
processed_dataset = preprocess_data(dataset)

# 定義你的訓練參數
training_args = {
    "num_train_epochs": 3,   # 訓練週期數
    "per_device_train_batch_size": 8,  # 每個設備的訓練批次大小
    "save_steps": 10_000,    # 每10,000步保存一次檢查點
    "weight_decay": 0.01,    # 權重衰減的強度
    # 根據需要添加更多參數
}

# 初始化 Trainer
from transformers import Trainer, TrainingArguments

training_arguments = TrainingArguments(**training_args)
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=processed_dataset,
    # 根據需要添加更多參數
)

# 開始微調
trainer.train()

# 保存微調後的模型
model.save_pretrained("path_to_save_your_model")
```
:::

#### Using GPU Code
:::spoiler Using GPU
```python=
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch

# 檢查是否有可用的 CUDA (GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 指定本地模型的路徑
local_model_path = "FT_1226/model"  # 替換為你本地模型的路徑

# 載入本地的模型和分詞器
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# 為微調準備你的數據集
# 這應該是包含你希望模型學習的文本的數據集
dataset = "FT_1226/data/pythonTrainingData_w3schools.txt"  # 替換為你數據集的路徑

# 定義一個方法來預處理數據
def preprocess_data(data):
    # 在這裡實現預處理步驟，如分詞等
    pass

# 預處理你的數據集
processed_dataset = preprocess_data(dataset)

# 定義你的訓練參數
# 定義訓練參數，並在其中指定使用 GPU
training_args = TrainingArguments(
    output_dir="FT_1226/model_fine_tuned",  # 模型和訓練日誌的輸出目錄
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    weight_decay=0.01,
    # 根據需要添加更多參數
    device=device  # 指定使用的設備
)

# 初始化 Trainer
from transformers import Trainer, TrainingArguments

training_arguments = TrainingArguments(**training_args)
# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    # 根據需要添加更多參數
)

# 開始微調
trainer.train()

# 保存微調後的模型
model.save_pretrained("FT_1226/model_fine_tuned")
```
:::
#### Error
:::spoiler Error - 1
HTTPError                                 Traceback (most recent call last)
File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\huggingface_hub\utils\_errors.py:270, in hf_raise_for_status(response, endpoint_name)
    269 try:
--> 270     response.raise_for_status()
    271 except HTTPError as e:

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\requests\models.py:1021, in Response.raise_for_status(self)
   1020 if http_error_msg:
-> 1021     raise HTTPError(http_error_msg, response=self)

HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/FT_1226/model/resolve/main/tokenizer_config.json

The above exception was the direct cause of the following exception:

RepositoryNotFoundError                   Traceback (most recent call last)
File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\transformers\utils\hub.py:389, in cached_file(path_or_repo_id, filename, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    387 try:
    388     # Load from URL or cache if already cached
--> 389     resolved_file = hf_hub_download(
    390         path_or_repo_id,
    391         filename,
    392         subfolder=None if len(subfolder) == 0 else subfolder,
    393         repo_type=repo_type,
    394         revision=revision,
...
    420         f"'https://huggingface.co/{path_or_repo_id}' for available revisions."
    421     ) from e

OSError: FT_1226/model is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
:::
:::spoiler Solve The Error - 1 Code
```python=
import os

# 检查路径是否存在
local_model_path = "FT_1226/model"  # 确保这是正确的路径
if not os.path.exists(local_model_path):
    print(f"路径不存在: {local_model_path}")
else:
    print(f"找到模型路径: {local_model_path}")
    # 检查必要的文件是否在路径中
    required_files = ['pytorch_model.bin', 'config.json']
    missing_files = [f for f in required_files if not os.path.isfile(os.path.join(local_model_path, f))]
    if missing_files:
        print(f"缺失文件: {missing_files}")
    else:
        print("所有必要的模型文件都已找到")
```
:::

:::spoiler Error - 2
HTTPError                                 Traceback (most recent call last)
File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\huggingface_hub\utils\_errors.py:270, in hf_raise_for_status(response, endpoint_name)
    269 try:
--> 270     response.raise_for_status()
    271 except HTTPError as e:

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\requests\models.py:1021, in Response.raise_for_status(self)
   1020 if http_error_msg:
-> 1021     raise HTTPError(http_error_msg, response=self)

HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/FT_1226/model/resolve/main/tokenizer_config.json

The above exception was the direct cause of the following exception:

RepositoryNotFoundError                   Traceback (most recent call last)
File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\transformers\utils\hub.py:389, in cached_file(path_or_repo_id, filename, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    387 try:
    388     # Load from URL or cache if already cached
--> 389     resolved_file = hf_hub_download(
    390         path_or_repo_id,
    391         filename,
    392         subfolder=None if len(subfolder) == 0 else subfolder,
    393         repo_type=repo_type,
    394         revision=revision,
...
    420         f"'https://huggingface.co/{path_or_repo_id}' for available revisions."
    421     ) from e

OSError: FT_1226/model is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...  
:::

### Test at home
:::spoiler Error - 3
output:
C:\Users\mason\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
WARNING:tensorflow:From C:\Users\mason\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

Using device: cuda
Some weights of LlamaForCausalLM were not initialized from the model checkpoint at FT_1226/model and are newly initialized: 
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.  

Error:  
TypeError                                 Traceback (most recent call last)
Cell In[1], line 29
     25 processed_dataset = preprocess_data(dataset)
     27 # 定義你的訓練參數
     28 # 定義訓練參數，並在其中指定使用 GPU
---> 29 training_args = TrainingArguments(
     30     output_dir="FT_1226/model_fine_tuned",  # 模型和訓練日誌的輸出目錄
     31     num_train_epochs=3,
     32     per_device_train_batch_size=8,
     33     save_steps=10_000,
     34     weight_decay=0.01,
     35     # 根據需要添加更多參數
     36     device=device  # 指定使用的設備
     37 )
     39 # 初始化 Trainer
     40 from transformers import Trainer, TrainingArguments

TypeError: TrainingArguments.\__init__() got an unexpected keyword argument 'device'  

:::

:::spoiler Error - 4

KeyboardInterrupt                         Traceback (most recent call last)
Cell In[1], line 13
     11 # 載入本地的模型和分詞器
     12 tokenizer = AutoTokenizer.from_pretrained(local_model_path)
---> 13 model = AutoModelForCausalLM.from_pretrained(local_model_path)
     15 # 為微調準備你的數據集
     16 # 這應該是包含你希望模型學習的文本的數據集
     17 dataset = "FT_1226/data/pythonTrainingData_w3schools.txt"  # 替換為你數據集的路徑

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\transformers\models\auto\auto_factory.py:566, in _BaseAutoModelClass.from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    564 elif type(config) in cls._model_mapping.keys():
    565     model_class = _get_model_class(config, cls._model_mapping)
--> 566     return model_class.from_pretrained(
    567         pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
    568     )
    569 raise ValueError(
    570     f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
    571     f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
    572 )

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\transformers\modeling_utils.py:3480, in PreTrainedModel.from_pretrained(cls, pretrained_model_name_or_path, config, cache_dir, ignore_mismatched_sizes, force_download, local_files_only, token, revision, use_safetensors, *model_args, **kwargs)
   3471     if dtype_orig is not None:
   3472         torch.set_default_dtype(dtype_orig)
   3473     (
   3474         model,
   3475         missing_keys,
   3476         unexpected_keys,
   3477         mismatched_keys,
   3478         offload_index,
   3479         error_msgs,
-> 3480     ) = cls._load_pretrained_model(
   3481         model,
   3482         state_dict,
   3483         loaded_state_dict_keys,  # XXX: rename?
   3484         resolved_archive_file,
   3485         pretrained_model_name_or_path,
   3486         ignore_mismatched_sizes=ignore_mismatched_sizes,
   3487         sharded_metadata=sharded_metadata,
   3488         _fast_init=_fast_init,
   3489         low_cpu_mem_usage=low_cpu_mem_usage,
   3490         device_map=device_map,
   3491         offload_folder=offload_folder,
   3492         offload_state_dict=offload_state_dict,
   3493         dtype=torch_dtype,
   3494         is_quantized=(getattr(model, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES),
   3495         keep_in_fp32_modules=keep_in_fp32_modules,
   3496     )
   3498 model.is_loaded_in_4bit = load_in_4bit
   3499 model.is_loaded_in_8bit = load_in_8bit

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\transformers\modeling_utils.py:3734, in PreTrainedModel._load_pretrained_model(cls, model, state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, ignore_mismatched_sizes, sharded_metadata, _fast_init, low_cpu_mem_usage, device_map, offload_folder, offload_state_dict, dtype, is_quantized, keep_in_fp32_modules)
   3732     set_initialized_submodules(model, _loaded_keys)
   3733     # This will only initialize submodules that are not marked as initialized by the line above.
-> 3734     model.apply(model._initialize_weights)
   3736 # Set some modules to fp32 if any
   3737 if keep_in_fp32_modules is not None:

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\torch\nn\modules\module.py:897, in Module.apply(self, fn)
    862 r"""Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
    863 as well as self. Typical use includes initializing the parameters of a model
    864 (see also :ref:`nn-init-doc`).
   (...)
    894 
    895 """
    896 for module in self.children():
--> 897     module.apply(fn)
    898 fn(self)
    899 return self

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\torch\nn\modules\module.py:897, in Module.apply(self, fn)
    862 r"""Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
    863 as well as self. Typical use includes initializing the parameters of a model
    864 (see also :ref:`nn-init-doc`).
   (...)
    894 
    895 """
    896 for module in self.children():
--> 897     module.apply(fn)
    898 fn(self)
    899 return self

    [... skipping similar frames: Module.apply at line 897 (2 times)]

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\torch\nn\modules\module.py:897, in Module.apply(self, fn)
    862 r"""Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
    863 as well as self. Typical use includes initializing the parameters of a model
    864 (see also :ref:`nn-init-doc`).
   (...)
    894 
    895 """
    896 for module in self.children():
--> 897     module.apply(fn)
    898 fn(self)
    899 return self

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\torch\nn\modules\module.py:898, in Module.apply(self, fn)
    896 for module in self.children():
    897     module.apply(fn)
--> 898 fn(self)
    899 return self

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\transformers\modeling_utils.py:1390, in PreTrainedModel._initialize_weights(self, module)
   1388 if getattr(module, "_is_hf_initialized", False):
   1389     return
-> 1390 self._init_weights(module)
   1391 module._is_hf_initialized = True

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\transformers\models\llama\modeling_llama.py:732, in LlamaPreTrainedModel._init_weights(self, module)
    730 std = self.config.initializer_range
    731 if isinstance(module, nn.Linear):
--> 732     module.weight.data.normal_(mean=0.0, std=std)
    733     if module.bias is not None:
    734         module.bias.data.zero_()

KeyboardInterrupt: 
:::spoiler Error - 4
---------------------------------------------------------------------------
KeyboardInterrupt                         Traceback (most recent call last)
Cell In[1], line 13
     11 # 載入本地的模型和分詞器
     12 tokenizer = AutoTokenizer.from_pretrained(local_model_path)
---> 13 model = AutoModelForCausalLM.from_pretrained(local_model_path)
     15 # 為微調準備你的數據集
     16 # 這應該是包含你希望模型學習的文本的數據集
     17 dataset = "FT_1226/data/pythonTrainingData_w3schools.txt"  # 替換為你數據集的路徑

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\transformers\models\auto\auto_factory.py:566, in _BaseAutoModelClass.from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    564 elif type(config) in cls._model_mapping.keys():
    565     model_class = _get_model_class(config, cls._model_mapping)
--> 566     return model_class.from_pretrained(
    567         pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
    568     )
    569 raise ValueError(
    570     f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
    571     f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
    572 )

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\transformers\modeling_utils.py:3480, in PreTrainedModel.from_pretrained(cls, pretrained_model_name_or_path, config, cache_dir, ignore_mismatched_sizes, force_download, local_files_only, token, revision, use_safetensors, *model_args, **kwargs)
   3471     if dtype_orig is not None:
   3472         torch.set_default_dtype(dtype_orig)
   3473     (
   3474         model,
   3475         missing_keys,
   3476         unexpected_keys,
   3477         mismatched_keys,
   3478         offload_index,
   3479         error_msgs,
-> 3480     ) = cls._load_pretrained_model(
   3481         model,
   3482         state_dict,
   3483         loaded_state_dict_keys,  # XXX: rename?
   3484         resolved_archive_file,
   3485         pretrained_model_name_or_path,
   3486         ignore_mismatched_sizes=ignore_mismatched_sizes,
   3487         sharded_metadata=sharded_metadata,
   3488         _fast_init=_fast_init,
   3489         low_cpu_mem_usage=low_cpu_mem_usage,
   3490         device_map=device_map,
   3491         offload_folder=offload_folder,
   3492         offload_state_dict=offload_state_dict,
   3493         dtype=torch_dtype,
   3494         is_quantized=(getattr(model, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES),
   3495         keep_in_fp32_modules=keep_in_fp32_modules,
   3496     )
   3498 model.is_loaded_in_4bit = load_in_4bit
   3499 model.is_loaded_in_8bit = load_in_8bit

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\transformers\modeling_utils.py:3734, in PreTrainedModel._load_pretrained_model(cls, model, state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, ignore_mismatched_sizes, sharded_metadata, _fast_init, low_cpu_mem_usage, device_map, offload_folder, offload_state_dict, dtype, is_quantized, keep_in_fp32_modules)
   3732     set_initialized_submodules(model, _loaded_keys)
   3733     # This will only initialize submodules that are not marked as initialized by the line above.
-> 3734     model.apply(model._initialize_weights)
   3736 # Set some modules to fp32 if any
   3737 if keep_in_fp32_modules is not None:

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\torch\nn\modules\module.py:897, in Module.apply(self, fn)
    862 r"""Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
    863 as well as self. Typical use includes initializing the parameters of a model
    864 (see also :ref:`nn-init-doc`).
   (...)
    894 
    895 """
    896 for module in self.children():
--> 897     module.apply(fn)
    898 fn(self)
    899 return self

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\torch\nn\modules\module.py:897, in Module.apply(self, fn)
    862 r"""Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
    863 as well as self. Typical use includes initializing the parameters of a model
    864 (see also :ref:`nn-init-doc`).
   (...)
    894 
    895 """
    896 for module in self.children():
--> 897     module.apply(fn)
    898 fn(self)
    899 return self

    [... skipping similar frames: Module.apply at line 897 (2 times)]

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\torch\nn\modules\module.py:897, in Module.apply(self, fn)
    862 r"""Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
    863 as well as self. Typical use includes initializing the parameters of a model
    864 (see also :ref:`nn-init-doc`).
   (...)
    894 
    895 """
    896 for module in self.children():
--> 897     module.apply(fn)
    898 fn(self)
    899 return self

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\torch\nn\modules\module.py:898, in Module.apply(self, fn)
    896 for module in self.children():
    897     module.apply(fn)
--> 898 fn(self)
    899 return self

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\transformers\modeling_utils.py:1390, in PreTrainedModel._initialize_weights(self, module)
   1388 if getattr(module, "_is_hf_initialized", False):
   1389     return
-> 1390 self._init_weights(module)
   1391 module._is_hf_initialized = True

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\transformers\models\llama\modeling_llama.py:732, in LlamaPreTrainedModel._init_weights(self, module)
    730 std = self.config.initializer_range
    731 if isinstance(module, nn.Linear):
--> 732     module.weight.data.normal_(mean=0.0, std=std)
    733     if module.bias is not None:
    734         module.bias.data.zero_()

KeyboardInterrupt: 
:::

## 12/27｜Sprint Week Day 2
### 方向調整

My grandmother is an illiterate farmer, and I hope to let her know that her vegetable sales are increasing every year. In 2019, she sold 100 kg, in 2020 she sold 120 kg, in 2021 she sold 140 kg, in 2022 she sold 160 kg, and in 2023 she sold 200 kg.Can you give me the code to generate the chart which title name Vegetable Sales Over Time?